import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# =============================================================
# 1. БАЗОВЫЕ СТОП-СЛОВА (общие для всех текстов)
# =============================================================

BASE_STOPWORDS = {
    # русские служебные слова
    "и", "в", "во", "на", "не", "что", "он", "она", "оно", "они",
    "я", "ты", "вы", "мы",
    "а", "но", "как", "так", "же",
    "с", "со", "к", "ко", "у", "от", "до", "по", "из", "за", "над", "под",
    "это", "тогда", "там", "тут", "здесь",
    "его", "ее", "их", "ему", "ей", "им",
    "бы", "ли", "то", "вот", "уж", "ну",
    "для", "при", "без", "об", "про", "надо",
    "да", "или", "если", "чтобы",
    "о", "об", "был", "было", "были",
    "ещё", "еще", "уже", "только", "всё", "все","рис",

    # английские служебные слова
    "in", "on", "to", "and", "the", "is", "of", "for", "from", "by",
}

# =============================================================
# 2. ИНДИВИДУАЛЬНЫЕ СТОП-СЛОВА ДЛЯ КАЖДОГО ДИПЛОМА
#    (ключ — имя файла в папке corpus, ПОЛНОСТЬЮ как в названии)
# =============================================================

PERSONAL_STOP = {

    # --- arina.txt ---
    "arina.txt": {
        "i", "j", "k", "double", "float"
    },

    # --- Fedorov_MV.txt ---
    "Fedorov_MV.txt": {
        # единицы измерения, обозначения и т.п.
        "м", "н", "ф", "кг", "мм", "см", "мпа", "м2", "м3",
        # технические сокращения
        "расч", "форм", "табл", "рис",
    },

    # --- kolesnikova_ds.txt ---
    "kolesnikova_ds.txt": {
        "self", "score", "z", "x", "y",
        "рисунок", "табл", "png", "jpg","ктр",
    },

    # --- kucheryavaya_mi.txt ---
    "kucheryavaya_mi.txt": {
        "float", "vec3", "gl", "shader",
        "f", "t", "i", "e",
    },

    # --- kuznetsova_av.txt ---
    "kuznetsova_av.txt": {
        "service", "desk", "rpa", "naumen", "itsm",
    },

    # --- Miroshnichenko_DA.txt ---
    "Miroshnichenko_DA.txt": {
        # много художественных служебных слов и имён
        "бывало", "только", "ещё", "еще",
        "было", "был", "была",
        "степан", "трофимович",
    },

    # --- ozhegova_ea_VKR.txt (Fortran-код) ---
    "ozhegova_ea_VKR.txt": {
        # служебные слова Fortran
        "do", "enddo", "integer", "subroutine", "intent",
        "real", "double", "module", "contains", "use",
        "endif", "if", "then", "else",
        # имена переменных
        "l1", "l2", "x1", "inz", "tmp", "arr",
    },

    # --- Patalakha_ad_VKR.txt ---
    "Patalakha_ad_VKR.txt": {
        "кв", "вкр", "глава", "табл", "рис",
    },

    # --- Sokolova_IP.txt ---
    "Sokolova_IP.txt": {
        "бд", "sql", "select", "from", "join",
        "табл", "рис", "json","быть","может",
    },

    # --- Valchuk_LV.txt ---
    "Valchuk_LV.txt": {
        "ооо", "зао", "оао", "агроторг", "ооо", "ooo",
        "табл", "рис","также",
    },

    # --- vilydanova_la.txt ---
    "vilydanova_la.txt": {
        "пао", "оао", "сургутнефтегаз", "роснефть", "газпром",
        "лукойл", "нефтегаз", "табл", "рис",
        "акционерное", "общество",
    },
}

# =============================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================

def read_text_safely(path: Path) -> str:
    """
    Пытаемся прочитать текст в одной из распространённых кодировок.
    """
    for enc in ("utf-8", "cp1251", "windows-1251"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # если вообще не получилось, пусть явно упадёт
    return path.read_text(encoding="utf-8")


def get_words(text: str, filename: str) -> list[str]:
    """
    Разбивает текст на слова, приводит к нижнему регистру
    и фильтрует:
      - базовые стоп-слова;
      - персональные стоп-слова для данного файла;
      - чистые числа;
      - односимвольные токены;
      - короткие латинские "технические" сокращения.
    """
    # регулярка: последовательности букв/цифр (включая кириллицу)
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    # берём персональный список для файла (если есть)
    personal = PERSONAL_STOP.get(filename, set())

    # общий стоп-лист
    stop = set(BASE_STOPWORDS) | set(personal)

    cleaned: list[str] = []
    for w in words:
        # 1) стоп-слова
        if w in stop:
            continue

        # 2) чистые числа
        if w.isdigit():
            continue

        # 3) односимвольные токены (любой алфавит)
        if len(w) == 1:
            continue

        # 4) короткие латинские технические аббревиатуры (2-3 символа)
        if len(w) <= 3 and not ('а' <= w[0] <= 'я'):
            # если начинается НЕ с русской буквы — считаем тех. токеном
            continue

        cleaned.append(w)

    return cleaned


def analyze_text(words: list[str], top_n: int = 200) -> dict:
    """
    Основной расчёт параметров закона Ципфа для списка слов.
    """
    total_words = len(words)
    counter = Counter(words)
    sorted_items = counter.most_common()

    if top_n:
        sorted_items = sorted_items[:top_n]

    ranks = []
    freqs_rel = []
    freqs_theor = []
    const_fr = []

    # экспериментальные данные
    for rank, (word, freq) in enumerate(sorted_items, start=1):
        f_rel = freq / total_words
        ranks.append(rank)
        freqs_rel.append(f_rel)
        const_fr.append(f_rel * rank)

    # средняя константа ⟨F_r * r⟩
    C_mean = sum(const_fr) / len(const_fr)

    # оптимальная константа C* по МНК:
    # C* = (Σ f_exp(r)/r) / (Σ 1/r^2)
    num = sum(f / r for f, r in zip(freqs_rel, ranks))
    denom = sum(1 / (r * r) for r in ranks)
    C_opt = num / denom

    # теоретические частоты
    for r in ranks:
        freqs_theor.append(C_opt / r)

    # среднеквадратичное отклонение
    mse = sum((f - t) ** 2 for f, t in zip(freqs_rel, freqs_theor)) / len(ranks)

    return {
        "total_words": total_words,
        "unique_words": len(counter),
        "ranks": ranks,
        "freqs_rel": freqs_rel,
        "freqs_theor": freqs_theor,
        "C_mean": C_mean,
        "C_opt": C_opt,
        "mse": mse,
        "top_list": sorted_items[:20],
    }


def plot_zipf(result: dict, title: str = "") -> None:
    """
    Рисует экспериментальную и теоретическую Zipf-кривые в лог–лог шкалах.
    """
    plt.figure()
    plt.loglog(result["ranks"], result["freqs_rel"], "o", label="Эксперимент")
    plt.loglog(result["ranks"], result["freqs_theor"], "-", label=f"Теория: C={result['C_opt']:.4f}")
    plt.title(title)
    plt.xlabel("Ранг слова (log r)")
    plt.ylabel("Относительная частота (log f)")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================
# 4. АНАЛИЗ КОРПУСА
# =============================================================

def analyze_corpus(folder: str, top_n: int = 200) -> dict:
    """
    Анализирует все .txt-файлы в папке и печатает сводку.
    """
    folder_path = Path(folder)
    results: dict[str, dict] = {}

    for path in sorted(folder_path.glob("*.txt")):
        print(f"\n=== Файл: {path.name} ===")
        text = read_text_safely(path)
        words = get_words(text, filename=path.name)
        res = analyze_text(words, top_n=top_n)
        results[path.name] = res

        print(f"Всего слов: {res['total_words']}")
        print(f"Уникальных слов: {res['unique_words']}")
        print(f"Средняя константа ⟨F_r * r⟩: {res['C_mean']:.4f}")
        print(f"Оптимальная константа C*: {res['C_opt']:.4f}")
        print(f"MSE: {res['mse']:.6e}")
        print("\nТоп-10 слов:")
        for i, (w, f) in enumerate(res["top_list"][:10], start=1):
            print(f"{i:2d}. {w:20s} {f}")

    return results


def compare_two_files(folder: str, file1: str, file2: str, top_n: int = 200) -> None:
    """
    Сравнение двух конкретных дипломов (например, Фёдорова и Колесниковой).
    """
    folder_path = Path(folder)

    text1 = read_text_safely(folder_path / file1)
    text2 = read_text_safely(folder_path / file2)

    words1 = get_words(text1, filename=file1)
    words2 = get_words(text2, filename=file2)

    res1 = analyze_text(words1, top_n=top_n)
    res2 = analyze_text(words2, top_n=top_n)

    print(f"\nСравнение файлов {file1} и {file2}")
    print("-" * 60)
    print(f"{file1}:  ⟨F_r r⟩ = {res1['C_mean']:.4f},  C* = {res1['C_opt']:.4f},  MSE = {res1['mse']:.6e}")
    print(f"{file2}:  ⟨F_r r⟩ = {res2['C_mean']:.4f},  C* = {res2['C_opt']:.4f},  MSE = {res2['mse']:.6e}")

    plot_zipf(res1, title=f"Закон Ципфа: {file1}")
    plot_zipf(res2, title=f"Закон Ципфа: {file2}")

# =============================================================
# 5. ТОЧКА ВХОДА
# =============================================================

if __name__ == "__main__":
    corpus_dir = "corpus"

    # 1) общий анализ всего корпуса
    analyze_corpus(corpus_dir, top_n=200)

    # 2) сравнение двух дипломов (Фёдоров и Колесникова)
    compare_two_files(corpus_dir, "Fedorov_MV.txt", "kolesnikova_ds.txt", top_n=200)
