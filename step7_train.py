# Шаг 7. Обучаем логистическую регрессию: одна модель "да/нет" на каждую из 19 меток.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
labels = train.columns[3:]

# TF-IDF так же, как в шаге 6
vectorizer = TfidfVectorizer(max_features=10000, lowercase=True)
X_train = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])
words = vectorizer.get_feature_names_out()

# Сюда складываем предсказания 0/1 для test
predictions = pd.DataFrame()
predictions["text"] = test["text"]

for label in labels:
    # max_iter=1000 - даём модели достаточно шагов, чтобы подобрать веса
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train[label])
    predictions[label] = model.predict(X_test)

    # 5 слов с самым большим весом - они сильнее всего "за" эту метку
    top = model.coef_[0].argsort()[::-1][:5]
    print(f"{label}: {', '.join(words[i] for i in top)}")

predictions.to_csv("data/predictions.csv", index=False)
print("\nСохранено: data/predictions.csv", predictions.shape)
