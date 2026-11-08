# сравниваем несколько моделей на одних и тех же TF-IDF признаках.
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
labels = train.columns[3:-1]  # 19 меток (последняя колонка text_clean - не метка)

# TF-IDF так же, как в шагах 6-10
vectorizer = TfidfVectorizer(max_features=10000, lowercase=True)
X_train = vectorizer.fit_transform(train["text_clean"])
X_test = vectorizer.transform(test["text_clean"])

# модели для сравнения. Для каждой метки обучается своя копия модели.
models = {
    "Логистическая регрессия": lambda: LogisticRegression(max_iter=1000),
    "Логистическая регрессия (balanced)": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Наивный Байес": lambda: MultinomialNB(),
    "Линейный SVM": lambda: LinearSVC(),
    "Линейный SVM (balanced)": lambda: LinearSVC(class_weight="balanced"),
}

rows = []
for name, make_model in models.items():
    start = time.time()
    p, r, f = [], [], []
    for label in labels:
        model = make_model()
        model.fit(X_train, train[label])
        pred = model.predict(X_test)
        p.append(precision_score(test[label], pred, zero_division=0))
        r.append(recall_score(test[label], pred, zero_division=0))
        f.append(f1_score(test[label], pred, zero_division=0))
    rows.append({
        "Модель": name,
        "macro-P": sum(p) / len(labels),
        "macro-R": sum(r) / len(labels),
        "macro-F1": sum(f) / len(labels),
        "Время, с": round(time.time() - start, 1),
    })
    print(name, "- готово")

result = pd.DataFrame(rows).sort_values("macro-F1", ascending=False)
print("\nСравнение моделей на test:")
print(result.round(3).to_string(index=False))

# график: macro-F1 каждой модели
plt.figure(figsize=(9, 4))
plt.barh(result["Модель"], result["macro-F1"])
plt.gca().invert_yaxis()  # лучшая модель сверху
for i, v in enumerate(result["macro-F1"]):
    plt.text(v + 0.005, i, f"{v:.3f}", va="center")  # подпись у столбика
plt.xlabel("macro-F1 на test")
plt.title("Сравнение моделей")
plt.xlim(0, max(result["macro-F1"]) + 0.08)
plt.tight_layout()
plt.savefig("images/models_compare.png")
print("\nГрафик сохранён: images/models_compare.png")
