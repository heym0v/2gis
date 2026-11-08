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

# группы меток по смыслу (в данных 2ГИС групп нет, мы их придумали сами)
groups = {
    "Цель визита": ["Просто поесть", "Расслабиться", "Потусить", "Фоткаться",
                    "Покормить детей", "Поболтать", "Поработать"],
    "Обстановка": ["Удобные сиденья", "Стильный интерьер", "Красивый вид",
                   "Просторный зал", "Добавить мистики"],
    "Еда и напитки": ["Необычные вкусы", "Большой бар", "Здоровое меню", "Кухни мира"],
    "Событие": ["Праздник", "Свидание", "День рождения"],
}
colors = {"Цель визита": "tab:blue", "Обстановка": "tab:green",
          "Еда и напитки": "tab:orange", "Событие": "tab:red"}

print("\nОтзывов по группам (хотя бы одна метка группы):")
for group, group_labels in groups.items():
    n = (df[group_labels].sum(axis=1) > 0).sum()
    print(f"{group}: {n}")

# график 1: столбики по меткам, метки идут по группам,
# внутри группы - от частой к редкой, у каждой группы свой цвет
names = []
values = []
bar_colors = []
for group, group_labels in groups.items():
    group_counts = counts[group_labels].sort_values(ascending=False)
    for label in group_counts.index:
        names.append(label)
        values.append(group_counts[label])
        bar_colors.append(colors[group])

plt.figure(figsize=(10, 7))
plt.barh(names, values, color=bar_colors)
plt.gca().invert_yaxis()  # первая группа сверху
# легенда: по одному цветному прямоугольнику на группу
for group, color in colors.items():
    plt.bar(0, 0, color=color, label=group)
plt.legend()
plt.xlabel("Количество отзывов")
plt.title("Сколько отзывов у каждой метки (по группам)")
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

# график 3: сколько меток у одного отзыва (в процентах)
per_review_percent = labels_per_review.value_counts(normalize=True).sort_index() * 100
plt.figure(figsize=(8, 5))
plt.bar(per_review_percent.index, per_review_percent.values)
for x, y in zip(per_review_percent.index, per_review_percent.values):
    plt.text(x, y + 1, f"{y:.1f}%", ha="center")  # подпись над столбиком
plt.xlabel("Сколько меток у отзыва")
plt.ylabel("Доля отзывов, %")
plt.title("Распределение отзывов по числу меток")
plt.tight_layout()
plt.savefig("images/labels_per_review.png")

# 4. сколько отзывов у одного заведения
reviews_per_branch = df["branch_id"].value_counts()
print("\nЗаведений с отзывами:", len(reviews_per_branch))
print("Отзывов на заведение: мин", reviews_per_branch.min(),
      "| макс", reviews_per_branch.max(),
      "| среднее", round(reviews_per_branch.mean(), 1),
      "| медиана", reviews_per_branch.median())

# график 4: гистограмма "сколько отзывов у заведения"
plt.figure(figsize=(10, 5))
plt.hist(reviews_per_branch, bins=range(1, 96))
plt.xlabel("Отзывов у заведения")
plt.ylabel("Количество заведений")
plt.title("Распределение отзывов по заведениям")
plt.tight_layout()
plt.savefig("images/reviews_per_branch.png")

print("\nГрафики сохранены в папку images/")
