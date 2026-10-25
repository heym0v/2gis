# cчитаем простую статистику и рисуем два графика.
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/reviews.csv")

# первые 3 колонки - branch_id, text, rating. Остальные 19 - метки.
labels = df.columns[3:]
print("Всего отзывов:", len(df))

# 1. сколько отзывов у каждой метки и какой это процент
counts = df[labels].sum().sort_values(ascending=False)
percent = (counts / len(df) * 100).round(1)
print("\nОтзывов на метку:")
for label in counts.index:
    print(f"{label}: {counts[label]} ({percent[label]}%)")

# 2. сколько меток у одного отзыва
labels_per_review = df[labels].sum(axis=1)
print("\nСколько меток у отзыва -> сколько таких отзывов:")
print(labels_per_review.value_counts().sort_index())

# 3. длина текстов (в символах)
length = df["text"].str.len()
print("\nДлина текста: мин", length.min(), "| макс", length.max(),
      "| средняя", round(length.mean()), "| медиана", length.median())

# график 1: столбики по меткам
plt.figure(figsize=(10, 6))
plt.barh(counts.index, counts.values)
plt.gca().invert_yaxis()  # самая частая метка сверху
plt.xlabel("Количество отзывов")
plt.title("Сколько отзывов у каждой метки")
plt.tight_layout()
plt.savefig("images/labels_count.png")

# график 2: гистограмма длины текстов (до 1000 символов, иначе не видно)
plt.figure(figsize=(10, 5))
plt.hist(length[length <= 1000], bins=50)
plt.xlabel("Длина отзыва, символов")
plt.ylabel("Количество отзывов")
plt.title("Длина отзывов (показаны отзывы до 1000 символов)")
plt.tight_layout()
plt.savefig("images/text_length.png")

print("\nГрафики сохранены в папку images/")
