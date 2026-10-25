# превращаем тексты отзывов в числа методом TF-IDF.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# берём 10 000 самых частых слов. lowercase=True - "Вкусно" и "вкусно" одно слово.
vectorizer = TfidfVectorizer(max_features=10000, lowercase=True)

# fit_transform: запоминаем слова и их IDF только по train и сразу считаем TF-IDF.
X_train = vectorizer.fit_transform(train["text"])
# transform: для test используем уже запомненные слова (в test не подглядываем).
X_test = vectorizer.transform(test["text"])

print("Размер X_train:", X_train.shape)
print("Размер X_test:", X_test.shape)

# смотрим на один отзыв: какие слова получили самый большой TF-IDF
words = vectorizer.get_feature_names_out()
row = X_train[0].toarray()[0]
print("\nОтзыв:", train["text"][0][:300])
print("\nТоп-10 слов этого отзыва по TF-IDF:")
for i in row.argsort()[::-1][:10]:
    if row[i] > 0:
        print(f"{words[i]}: {row[i]:.3f}")
