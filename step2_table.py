# Шаг 2. Делаем таблицу: одна строка = один отзыв, 19 колонок с метками (0 или 1).
import json
import pandas as pd

with open("data/task_2_vibes.json", encoding="utf-8") as f:
    data = json.load(f)

# Собираем все названия меток
labels = set()
for place in data:
    for label in place["nlp"]["aspect"]:
        labels.add(label)
labels = sorted(labels)
print("Меток:", len(labels))

# Словарь: (id заведения, текст отзыва) -> информация об отзыве.
# Ключ из двух частей, потому что один и тот же текст может быть у разных заведений.
reviews = {}
for place in data:
    for label, texts in place["nlp"]["aspect"].items():
        for text in texts:
            key = (place["branch_id"], text)
            if key not in reviews:
                # Первый раз видим этот отзыв: создаём строку, все метки = 0
                row = {"branch_id": place["branch_id"], "text": text, "rating": place["rating"]}
                for l in labels:
                    row[l] = 0
                reviews[key] = row
            # Ставим 1 у той метки, под которой нашли отзыв
            reviews[key][label] = 1

# Превращаем словарь в таблицу pandas
df = pd.DataFrame(list(reviews.values()))
print("Строк (отзывов):", len(df))
print("Колонок:", len(df.columns))
print(df.head())

# Сохраняем в CSV
df.to_csv("data/reviews.csv", index=False)
print("Сохранено в data/reviews.csv")
