# дополнительная оценка: метрики на train и test, матрицы ошибок и ROC-кривые.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, roc_auc_score)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
labels = train.columns[3:-1]  # 19 меток (последняя колонка text_clean - не метка)

# TF-IDF так же, как в шагах 6-9
vectorizer = TfidfVectorizer(max_features=10000, lowercase=True)
X_train = vectorizer.fit_transform(train["text_clean"])
X_test = vectorizer.transform(test["text_clean"])

# ---- 1. метрики на train и на test для обеих моделей ----
# если на train намного лучше, чем на test, модель переобучилась
models = {"обычная": None, "balanced": "balanced"}
summary = []
balanced_models = {}  # сохраним сбалансированные модели для матриц и ROC

for name, weight in models.items():
    scores = {"train": {"P": [], "R": [], "F1": []}, "test": {"P": [], "R": [], "F1": []}}
    for label in labels:
        model = LogisticRegression(max_iter=1000, class_weight=weight)
        model.fit(X_train, train[label])
        if weight == "balanced":
            balanced_models[label] = model
        for part, X, y in [("train", X_train, train[label]), ("test", X_test, test[label])]:
            pred = model.predict(X)
            scores[part]["P"].append(precision_score(y, pred, zero_division=0))
            scores[part]["R"].append(recall_score(y, pred, zero_division=0))
            scores[part]["F1"].append(f1_score(y, pred, zero_division=0))
    for part in ["train", "test"]:
        summary.append({
            "Модель": name,
            "Выборка": part,
            "macro-P": sum(scores[part]["P"]) / len(labels),
            "macro-R": sum(scores[part]["R"]) / len(labels),
            "macro-F1": sum(scores[part]["F1"]) / len(labels),
        })

print("Средние по 19 меткам показатели на train и test:")
print(pd.DataFrame(summary).round(3).to_string(index=False))

# ---- 2. матрицы ошибок для 4 меток (сбалансированная модель, test) ----
# две лучшие и две худшие метки по F1 из шага 9
four = ["Стильный интерьер", "Просто поесть", "Свидание", "Добавить мистики"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, label in zip(axes, four):
    pred = balanced_models[label].predict(X_test)
    # строки - правильный ответ (0/1), столбцы - ответ модели (0/1)
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(test[label], pred)
    print(f"\n{label}: TN={cm[0, 0]} FP={cm[0, 1]} FN={cm[1, 0]} TP={cm[1, 1]}")
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            # тёмный фон - белый текст, светлый - чёрный
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=12)
    ax.set_xticks([0, 1], ["нет", "да"])
    ax.set_yticks([0, 1], ["нет", "да"])
    ax.set_xlabel("Ответ модели")
    ax.set_ylabel("Правильный ответ")
    ax.set_title(label)
plt.tight_layout()
plt.savefig("images/confusion_matrices.png")

# ---- 3. ROC-кривые и AUC для тех же 4 меток ----
# predict_proba даёт вероятность метки; ROC показывает, как меняются
# доля найденных (TPR) и доля ложных срабатываний (FPR) при разных порогах
plt.figure(figsize=(7, 7))
print("\nAUC на test:")
for label in four:
    proba = balanced_models[label].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(test[label], proba)
    auc = roc_auc_score(test[label], proba)
    print(f"{label}: {auc:.3f}")
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="случайное угадывание (AUC = 0.5)")
plt.xlabel("FPR (доля ложных срабатываний)")
plt.ylabel("TPR (доля найденных, recall)")
plt.title("ROC-кривые, сбалансированная модель, test")
plt.legend()
plt.tight_layout()
plt.savefig("images/roc_curves.png")

# AUC для всех 19 меток - в одну таблицу
auc_all = {label: roc_auc_score(test[label], balanced_models[label].predict_proba(X_test)[:, 1])
           for label in labels}
auc_all = pd.Series(auc_all).sort_values(ascending=False)
print("\nAUC по всем меткам:")
print(auc_all.round(3).to_string())
print("Средний AUC:", round(auc_all.mean(), 3))

print("\nГрафики сохранены: images/confusion_matrices.png, images/roc_curves.png")
