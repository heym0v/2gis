# предобработка текста: убираем стоп-слова и приводим слова к начальной форме
# (лемматизация). Результат кладём в новую колонку text_clean в train и test.
import re
import pandas as pd
import pymorphy3

morph = pymorphy3.MorphAnalyzer()

# стоп-слова определяем по части речи: местоимения, предлоги, союзы,
# частицы и междометия (теги библиотеки pymorphy3)
STOP_POS = {"NPRO", "PREP", "CONJ", "PRCL", "INTJ"}
# но "не" и "нет" оставляем: без них "не понравилось" станет "понравилось"
KEEP = {"не", "нет"}

# одно и то же слово встречается много раз, поэтому запоминаем
# уже разобранные слова в словаре, чтобы не разбирать их заново
cache = {}


def lemma(word):
    if word not in cache:
        # казахские слова pymorphy3 не знает и портит ("туған" -> "туғать"),
        # поэтому слова с казахскими буквами оставляем как есть
        if re.search("[әғқңөұүһі]", word):
            cache[word] = word
            return word
        p = morph.parse(word)[0]  # самый вероятный разбор слова
        if p.tag.POS in STOP_POS and word not in KEEP:
            cache[word] = ""  # стоп-слово - выбрасываем
        else:
            cache[word] = p.normal_form  # начальная форма: "видом" -> "вид"
    return cache[word]


def clean(text):
    # переводим в нижний регистр и оставляем только слова из букв
    # ([^\W\d_]+ = подряд идущие буквы любого алфавита, в том числе казахского;
    # знаки препинания, цифры и эмодзи пропадают)
    words = re.findall(r"[^\W\d_]+", text.lower())
    lemmas = [lemma(w) for w in words]
    # убираем пустые (стоп-слова) и слова из одной буквы
    return " ".join(w for w in lemmas if len(w) > 1)


for name in ["train", "test"]:
    df = pd.read_csv(f"data/{name}.csv")
    df["text_clean"] = df["text"].apply(clean)
    df.to_csv(f"data/{name}.csv", index=False)
    print(name, "- обработано отзывов:", len(df))

print("Разных слов до лемматизации:", len(cache))
print("Разных лемм после:", len(set(cache.values())))

# пример: один отзыв до и после
print("\nДо:   ", df["text"].iloc[0][:300])
print("После:", df["text_clean"].iloc[0][:300])
