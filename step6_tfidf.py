# превращаем тексты отзывов в числа методом TF-IDF.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# берём 10 000 самых частых лемм. Текст уже очищен в шаге 5b
# (нижний регистр, без стоп-слов, слова в начальной форме).
vectorizer = TfidfVectorizer(max_features=10000, lowercase=True)

# fit_transform: запоминаем слова и их IDF только по train и сразу считаем TF-IDF.
X_train = vectorizer.fit_transform(train["text_clean"])
# transform: для test используем уже запомненные слова (в test не подглядываем).
X_test = vectorizer.transform(test["text_clean"])

print("Размер X_train:", X_train.shape)
print("Размер X_test:", X_test.shape)

# смотрим на один отзыв: какие слова получили самый большой TF-IDF
words = vectorizer.get_feature_names_out()
row = X_train[0].toarray()[0]
print("\nОтзыв:", train["text"][0][:300])
print("После очистки:", train["text_clean"][0][:300])
print("\nТоп-10 слов этого отзыва по TF-IDF:")
for i in row.argsort()[::-1][:10]:
    if row[i] > 0:
        print(f"{words[i]}: {row[i]:.3f}")

# ---- графики слов с наибольшим TF-IDF ----
import matplotlib.pyplot as plt

# средний TF-IDF каждого слова по всем отзывам train
mean_all = X_train.mean(axis=0).A1  # .A1 превращает результат в обычный массив
top = mean_all.argsort()[::-1][:20]  # 20 слов с самым большим средним весом

plt.figure(figsize=(8, 7))
plt.barh([words[i] for i in top], mean_all[top])
plt.gca().invert_yaxis()  # самое весомое слово сверху
plt.xlabel("Средний TF-IDF")
plt.title("20 слов с наибольшим средним TF-IDF (все отзывы train)")
plt.tight_layout()
plt.savefig("images/top_words_all.png")

# то же самое для 4 меток: средний TF-IDF только по отзывам с этой меткой
four = ["Просто поесть", "Свидание", "Поработать", "Покормить детей"]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, label in zip(axes.flatten(), four):
    rows = (train[label] == 1).values  # отзывы, у которых есть эта метка
    mean_label = X_train[rows].mean(axis=0).A1
    top = mean_label.argsort()[::-1][:10]
    ax.barh([words[i] for i in top], mean_label[top])
    ax.invert_yaxis()
    ax.set_title(f"{label} ({rows.sum()} отзывов)")
    ax.set_xlabel("Средний TF-IDF")
plt.tight_layout()
plt.savefig("images/top_words_labels.png")
print("\nГрафики сохранены: images/top_words_all.png, images/top_words_labels.png")
