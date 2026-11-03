"""Сопоставление некорректных значений справочников с эталонными.

Комбинация алгоритмов: нормализация текста (регистр, ё/е, пробелы),
расстояние Левенштейна, коэффициент Жаккара по словам и fuzzy ratio.

Использование:
    python reference_v2.py incorrect.csv punishment_types.csv matched_output.csv

Либо через переменные окружения INPUT_FILE / REF_FILE / OUTPUT_FILE.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import unicodedata
from dataclasses import dataclass

from rapidfuzz import distance, fuzz, process

# Стоп-слова для Jaccard: служебные слова размывают сходство коротких фраз.
STOP_WORDS = frozenset({
    'за', 'в', 'во', 'и', 'а', 'не', 'на', 'по', 'с', 'со', 'к', 'у', 'о', 'об',
    'от', 'до', 'из', 'при', 'для', 'или', 'но', 'что', 'как', 'его', 'её', 'ее',
})


def normalize(s: str | None) -> str:
    """Приведение строки к виду, пригодному для сравнения.

    - strip + нижний регистр;
    - схлопывание повторяющихся пробелов;
    - ё -> е (частая причина "мнимых" расхождений в русских справочниках);
    - удаление диакритики (NFKD).

    ВАЖНО: буквы 'й'/'Й' намеренно НЕ транслируются в 'и' — в NFKD они
    раскладываются с combining breve, который при удалении портит слово
    ("разбойное" -> "разбоиное").
    """
    s = (s or '').strip().lower()
    if not s:
        return ''
    s = s.replace('ё', 'е')
    s = ''.join(
        c for c in unicodedata.normalize('NFKD', s)
        if not unicodedata.combining(c)
    )
    return re.sub(r'\s+', ' ', s)


def jaccard_words(s1: str, s2: str, *, stop_words: bool = True) -> float:
    """Коэффициент Жаккара по множествам слов (0.0–1.0)."""
    if not s1 or not s2:
        return 0.0

    set1 = set(s1.split())
    set2 = set(s2.split())

    if stop_words:
        # Не удаляем стоп-слова, если это оставит множество пустым.
        r1 = set1 - STOP_WORDS
        r2 = set2 - STOP_WORDS
        if r1 and r2:
            set1, set2 = r1, r2

    union = set1 | set2
    if not union:
        return 1.0
    return len(set1 & set2) / len(union)


@dataclass(frozen=True)
class RefEntry:
    original: str   # как в справочнике — именно это значение попадёт в ответ
    norm: str       # нормализованное значение — для сравнения


def read_csv_column(path: str, column: str) -> list[str]:
    """Читает одну колонку из CSV, пропуская пустые значения."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(
                f"В файле {path} нет колонки '{column}'. "
                f"Найдены колонки: {reader.fieldnames}"
            )
        return [row[column].strip() for row in reader if (row[column] or '').strip()]


def score_pair(source_norm: str, ref_norm: str) -> tuple[int, float, float]:
    """Возвращает (Левенштейн, fuzzy ratio, Jaccard) для нормализованных строк."""
    lev = distance.Levenshtein.distance(source_norm, ref_norm)
    ratio = fuzz.WRatio(source_norm, ref_norm) / 100.0  # нормализуем к 0..1
    jac = jaccard_words(source_norm, ref_norm)
    return lev, ratio, jac


def find_best_matches(
    input_path: str,
    ref_path: str,
    output_path: str,
    *,
    source_column: str = 'value',
    ref_column: str = 'correct_value',
    min_ratio: float = 0.5,
    top_n: int = 3,
) -> list[dict]:
    """Сопоставляет значения из input_path со справочником ref_path.

    Каждый кандидат получает composite score — взвешенную сумму трёх метрик,
    приведённых к шкале 0..1:
        score = 0.5 * WRatio + 0.3 * Jaccard + 0.2 * (1 - Левенштейн/max_len)

    Выбор только по Левенштейну (как в исходной версии) ненадёжен: он штрафует
    перефразированные длинные записи. Порог уверенности — min_ratio.
    """
    for path, name in [(input_path, 'некорректные значения'), (ref_path, 'справочник')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Не найден файл для {name}: {path}')

    sources = read_csv_column(input_path, source_column)
    refs = [RefEntry(original=r, norm=normalize(r)) for r in read_csv_column(ref_path, ref_column)]
    if not refs:
        raise ValueError(f'Справочник пуст: {ref_path} (колонка {ref_column!r})')

    ref_norms = [e.norm for e in refs]  # один раз для prefilter

    # Дедупликация источников: одинаковые строки считаем один раз.
    cache: dict[str, dict] = {}
    results: list[dict] = []

    for source_val in sources:
        if source_val in cache:
            results.append(dict(cache[source_val]))
            continue

        src_norm = normalize(source_val)

        # Пустые/бессмысленные значения (например, "," или пробелы) не матчим —
        # иначе rapidfuzz даст им высокий score на любом кандидате.
        if not re.sub(r'[^\wа-яё]', '', src_norm):
            row = {
                'source_value': source_val, 'best_match': '', 'match_found': 'no',
                'score': 0.0, 'levenshtein_distance': None, 'fuzzy_ratio': None,
                'jaccard_words': None, 'alternatives': '',
            }
            cache[source_val] = row
            results.append(dict(row))
            continue

        # Быстрый предварительный отбор (C++ внутри rapidfuzz): берём ~K
        # ближайших кандидатов по WRatio, точный скоринг считаем только по ним.
        # extract возвращает [(choice, score, index), ...]; индексы позволяют
        # вернуться к оригинальному написанию справочника.
        k = min(len(refs), max(10, top_n * 4))
        prefilter = process.extract(
            src_norm,
            ref_norms,
            scorer=fuzz.WRatio,
            limit=k,
        )

        scored = []
        for _choice, _wscore, idx in prefilter:
            entry = refs[idx]
            lev, ratio, jac = score_pair(src_norm, entry.norm)
            # Composite score: взвешенная сумма метрик (0..1).
            # WRatio устойчив к порядку слов и частичным совпадениям,
            # Jaccard хорошо работает для многословных фраз.
            score = 0.5 * ratio + 0.3 * jac + 0.2 * _lev_similarity(src_norm, entry.norm, lev)
            scored.append((score, lev, ratio, jac, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        best_score, best_lev, best_ratio, best_jac, best_entry = scored[0]
        confident = best_score >= min_ratio

        row = {
            'source_value': source_val,
            'best_match': best_entry.original if confident else '',
            'match_found': 'yes' if confident else 'no',
            'score': round(best_score, 3),
            'levenshtein_distance': best_lev,
            'fuzzy_ratio': round(best_ratio, 3),
            'jaccard_words': round(best_jac, 3),
            'alternatives': ' | '.join(
                e.original for _, _, _, _, e in scored[1:top_n] if e.original != best_entry.original
            ),
        }
        cache[source_val] = row
        results.append(dict(row))

    fieldnames = ['source_value', 'best_match', 'match_found', 'score',
                  'levenshtein_distance', 'fuzzy_ratio', 'jaccard_words', 'alternatives']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return results


def _lev_similarity(a: str, b: str, lev: int) -> float:
    """Похожесть по Левенштейну в шкале 0..1 (1 = идентичны)."""
    m = max(len(a), len(b))
    return 1.0 - lev / m if m else 1.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('input_file', nargs='?', default=os.environ.get('INPUT_FILE'),
                        help='CSV с некорректными значениями')
    parser.add_argument('ref_file', nargs='?', default=os.environ.get('REF_FILE'),
                        help='CSV с эталонным справочником')
    parser.add_argument('output_file', nargs='?', default=os.environ.get('OUTPUT_FILE', 'matched_output.csv'),
                        help='Куда сохранить результат (по умолчанию matched_output.csv)')
    parser.add_argument('--source-column', default='value')
    parser.add_argument('--ref-column', default='correct_value')
    parser.add_argument('--min-ratio', type=float, default=0.5,
                        help='Порог composite score для признания совпадения (0..1)')
    parser.add_argument('--top-n', type=int, default=3,
                        help='Сколько альтернатив сохранять в колонке alternatives')
    args = parser.parse_args(argv)
    if not args.input_file or not args.ref_file:
        parser.error('Укажите input_file и ref_file (аргументами или INPUT_FILE/REF_FILE)')
    return args


if __name__ == '__main__':
    args = parse_args()
    try:
        results = find_best_matches(
            args.input_file, args.ref_file, args.output_file,
            source_column=args.source_column,
            ref_column=args.ref_column,
            min_ratio=args.min_ratio,
            top_n=args.top_n,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        sys.exit(1)

    matched = sum(1 for r in results if r['match_found'] == 'yes')
    print(f'Готово: обработано {len(results)} значений, уверенных совпадений: {matched}. '
          f'Результат сохранён в: {args.output_file}')
