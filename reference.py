import argparse
import csv
import os
import sys
from rapidfuzz import distance, fuzz

def jaccard_words(s1: str, s2: str) -> float:
    """Коэффициент Жаккара по словам (0.0–1.0)."""
    if not s1 or not s2:
        return 0.0
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()

    set1 = set(s1.split())
    set2 = set(s2.split())

    if not set1 and not set2:
        return 1.0

    intersection = set1 & set2
    union = set1 | set2

    return len(intersection) / len(union) if union else 0.0

def find_best_matches(
        input_path: str,
        ref_path: str,
        output_path: str,
        *,
        source_column: str = 'value',
        ref_column: str = 'correct_value',
        min_ratio: float = 0.5,
        top_n: int = 3,
):
    # Проверка существования файлов
    for path, name in [(input_path, "некорректные значения"), (ref_path, "справочник")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Не найден файл для {name}: {path}")

    # Читаем некорректные значения
    with open(input_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        inputs = [row for row in reader]

    # Читаем справочник
    with open(ref_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        refs = [row[ref_column] for row in reader if row.get(ref_column)]

    if not refs:
        raise ValueError(f"В справочнике '{ref_path}' не найдено значений в колонке '{ref_column}'")

    results = []

    for row in inputs:
        source_val = (row.get(source_column) or '').strip()
        best_match = ''
        best_lev = 9999
        best_ratio = -1
        best_jacc = -1.0

        for ref in refs:
            lev = distance.Levenshtein.normalized_distance(source_val.lower(), ref.lower())
            ratio = fuzz.WRatio(source_val, ref) / 100.0
            jacc = jaccard_words(source_val, ref)

            # Логика выбора лучшего совпадения: сначала по Жаккару и fuzzy-ratio, затем Левенштейн
            score = max(ratio, jacc)
            best_score = max(best_ratio, best_jacc)
            if (score > best_score + 1e-9) or (abs(score - best_score) <= 1e-9 and lev < best_lev):
                best_lev = lev
                best_ratio = ratio
                best_jacc = jacc
                best_match = ref

        confident = best_ratio >= min_ratio and best_match != ''
        results.append({
            'source_value': source_val,
            'best_match': best_match if confident else '',
            'confidence': 'высокая' if best_ratio >= 0.85 else ('средняя' if confident else 'низкая'),
            'levenshtein_norm': round(best_lev, 3),
            'fuzzy_ratio': round(best_ratio, 3),
            'jaccard_words': round(best_jacc, 3),
        })

    fieldnames = ['source_value', 'best_match', 'confidence',
                  'levenshtein_norm', 'fuzzy_ratio', 'jaccard_words']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

if __name__ == '__main__':
    # --- НАСТРОЙКИ ПУТЕЙ ПО УМОЛЧАНИЮ: поменяй эти строки под себя ---
    # Пример для Windows: r"C:\Users\Name\Documents\incorrect.csv"
    # Пример для macOS/Linux: "/home/user/data/incorrect.csv"
    DEFAULT_INPUT_FILE = r"C:\Users\Vadim\Documents\Справочники\miv_public_person_condemnation_1.csv"  # файл с некорректными значениями
    DEFAULT_REF_FILE = r"C:\Users\Vadim\Documents\Справочники\punishment_types.csv"  # файл со справочником
    DEFAULT_OUTPUT_FILE = r"C:\Users\Vadim\Documents\Справочники\matched_output.csv"  # куда сохранить результат

    # Пути можно задавать тремя способами (приоритет сверху вниз):
    #   1. Аргументы командной строки: python reference.py input.csv ref.csv out.csv
    #   2. Переменные окружения: INPUT_FILE / REF_FILE / OUTPUT_FILE
    #   3. Значения DEFAULT_* выше
    parser = argparse.ArgumentParser(description='Сопоставление некорректных значений справочников с эталонными.')
    parser.add_argument('input_file', nargs='?', default=None, help='CSV с некорректными значениями')
    parser.add_argument('ref_file', nargs='?', default=None, help='CSV с эталонным справочником')
    parser.add_argument('output_file', nargs='?', default=None, help='Куда сохранить результат')
    parser.add_argument('--source-column', default='value',
                        help="Название колонки с некорректными значениями в input_file (по умолчанию: value)")
    parser.add_argument('--ref-column', default='correct_value',
                        help="Название колонки справочника в ref_file (по умолчанию: correct_value)")
    parser.add_argument('--min-ratio', type=float, default=0.5,
                        help="Минимальный порог похожести 0..1 для признания совпадения (по умолчанию: 0.5)")
    parser.add_argument('--top-n', type=int, default=3,
                        help="Число лучших кандидатов, рассматриваемых для каждого значения (по умолчанию: 3)")
    args = parser.parse_args()

    def resolve(cli_value, env_value, default_value):
        value = cli_value or env_value or default_value
        return os.path.expanduser(os.path.normpath(value)) if value else None

    input_file = resolve(args.input_file, os.environ.get('INPUT_FILE'), DEFAULT_INPUT_FILE)
    ref_file = resolve(args.ref_file, os.environ.get('REF_FILE'), DEFAULT_REF_FILE)
    output_file = resolve(args.output_file, os.environ.get('OUTPUT_FILE'), DEFAULT_OUTPUT_FILE) or 'matched_output.csv'

    missing = [name for name, val in [('input_file', input_file), ('ref_file', ref_file)] if not val]
    if missing:
        parser.error(f'Не заданы путь(и): {", ".join(missing)}')

    # Запуск
    try:
        find_best_matches(
            input_file, ref_file, output_file,
            source_column=args.source_column,
            ref_column=args.ref_column,
            min_ratio=args.min_ratio,
            top_n=args.top_n,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        sys.exit(1)
    print(f"Готово! Результат сохранён в: {output_file}")

    # Запуск
    find_best_matches(input_file, ref_file, output_file)
    print(f"Готово! Результат сохранён в: {output_file}")
