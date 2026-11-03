import csv
import os
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

def find_best_matches(input_path: str, ref_path: str, output_path: str):
    # Проверка существования файлов
    for path, name in [(input_path, "некорректные значения"), (ref_path, "справочник")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Не найден файл для {name}: {path}")

    # Читаем некорректные значения
    with open(input_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        inputs = [row for row in reader]

    # Читаем справочник
    with open(ref_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Предполагаем, что колонка называется correct_value; если иначе — поправь ниже
        refs = [row['correct_value'] for row in reader if row.get('correct_value')]

    results = []

    for row in inputs:
        source_val = row.get('value', '')
        best_match = ''
        best_lev = 9999
        best_ratio = -1
        best_jacc = -1.0

        for ref in refs:
            lev = distance.Levenshtein.distance(source_val.lower(), ref.lower())
            ratio = fuzz.ratio(source_val, ref)
            jacc = jaccard_words(source_val, ref)

            # Логика выбора лучшего совпадения
            if lev < best_lev or (lev == best_lev and ratio > best_ratio):
                best_lev = lev
                best_ratio = ratio
                best_jacc = jacc
                best_match = ref

        results.append({
            'source_value': source_val,
            'best_match': best_match,
            'levenshtein_distance': best_lev,
            'fuzzy_ratio': best_ratio,
            'jaccard_words': round(best_jacc, 3)
        })

    fieldnames = ['source_value', 'best_match', 'levenshtein_distance', 'fuzzy_ratio', 'jaccard_words']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

if __name__ == '__main__':
    # --- НАСТРОЙКИ ПУТЕЙ: поменяй эти строки под себя ---
    # Пример для Windows: r"C:\Users\Name\Documents\incorrect.csv"
    # Пример для macOS/Linux: "/home/user/data/incorrect.csv"

    input_file = r"C:\Users\Vadim\Documents\Справочники\miv_public_person_condemnation_1.csv" # файл с некорректными значениями
    ref_file   = r"C:\Users\Vadim\Documents\Справочники\punishment_types.csv"# файл со справочником
    output_file = r"C:\Users\Vadim\Documents\Справочники\matched_output.csv"  # куда сохранить результат

    # Запуск
    find_best_matches(input_file, ref_file, output_file)
    print(f"Готово! Результат сохранён в: {output_file}")
