import re
from collections import Counter
import matplotlib.pyplot as plt


def get_words(text: str):
    """
    Разбиваем текст на слова:
    - приводим к нижнему регистру,
    - берём только буквы/цифры (в т.ч. русские).
    """
    # \w с флагом re.UNICODE захватывает и русские буквы
    return re.findall(r'\w+', text.lower(), flags=re.UNICODE)


def zipf_constant(filename: str, top_n: int | None = None, show_plot: bool = True):
   
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

      words = get_words(text)
    total_words = len(words)
    print(f"Всего слов (с повторами): {total_words}")

       counter = Counter(words)
   
    sorted_words = counter.most_common()
    if top_n is not None:
        sorted_words = sorted_words[:top_n]
    
    constants = []
    ranks = []
    freqs = []

    print("\nТоп слов и значения C = f * r:")
    print("Ранг\tСлово\tЧастота\tC=f*r")

    for rank, (word, freq) in enumerate(sorted_words, start=1):
        C = freq * rank
        constants.append(C)
        ranks.append(rank)
        freqs.append(freq / total_words)  # относительная частота (для графика)
        if rank <= 20:  # чтобы не засорять вывод, печатаем первые 20
            print(f"{rank}\t{word}\t{freq}\t{C}")
   
    avg_C = sum(constants) / len(constants)
    print(f"\nОценка константы C по {len(constants)} словам: {avg_C:.2f}")
    
    if show_plot:
        plt.figure()
        plt.plot(ranks, freqs)
        plt.xlabel("Место слова в частотном словаре (ранг)")
        plt.ylabel("Частота встречаемости слова")
        plt.title("Проверка закона Ципфа")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
       zipf_constant("text.txt", top_n=200, show_plot=True)

