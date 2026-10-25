# борьба с дисбалансом: class_weight="balanced" и сравнение с моделью из шага 7.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
old = pd.read_csv("data/predictions.csv")  # предсказания обычной модели (шаг 7)
labels = train.columns[3:]

# TF-IDF так же, как в шагах 6 и 7
vectorizer = TfidfVectorizer(max_features=10000, lowercase=True)
X_train = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])

rows = []
for label in labels:
    # class_weight="balanced": ошибка на редком классе "1" стоит дороже,
    # поэтому модель перестаёт почти всегда отвечать "нет"
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, train[label])
    new = model.predict(X_test)

    y_true = test[label]
    rows.append({
        "Метка": label,
        "P было": precision_score(y_true, old[label], zero_division=0),
        "P стало": precision_score(y_true, new, zero_division=0),
        "R было": recall_score(y_true, old[label], zero_division=0),
        "R стало": recall_score(y_true, new, zero_division=0),
        "F1 было": f1_score(y_true, old[label], zero_division=0),
        "F1 стало": f1_score(y_true, new, zero_division=0),
    })

result = pd.DataFrame(rows).sort_values("F1 стало", ascending=False)
print(result.round(2).to_string(index=False))
print("\nmacro-F1 было:", round(result["F1 было"].mean(), 3))
print("macro-F1 стало:", round(result["F1 стало"].mean(), 3))

# график: F1 обычной и сбалансированной модели рядом
y = range(len(result))
plt.figure(figsize=(10, 7))
plt.barh([i - 0.2 for i in y], result["F1 было"], height=0.4, label="обычная")
plt.barh([i + 0.2 for i in y], result["F1 стало"], height=0.4, label='class_weight="balanced"')
plt.yticks(list(y), result["Метка"])
plt.gca().invert_yaxis()
plt.xlabel("F1")
plt.title("F1 по меткам: до и после балансировки классов")
plt.legend()
plt.tight_layout()
plt.savefig("images/f1_compare.png")
print("\nГрафик сохранён: images/f1_compare.png")
