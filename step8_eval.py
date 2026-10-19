# Шаг 8. Оцениваем качество модели на test: precision, recall и F1 для каждой метки.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

test = pd.read_csv("data/test.csv")
predictions = pd.read_csv("data/predictions.csv")
labels = test.columns[3:]

rows = []
for label in labels:
    y_true = test[label]          # правильные ответы
    y_pred = predictions[label]   # ответы модели
    rows.append({
        "Метка": label,
        "Отзывов в test": y_true.sum(),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    })

result = pd.DataFrame(rows).sort_values("F1", ascending=False)
print(result.round(2).to_string(index=False))
print("\nmacro-F1 (среднее F1 по 19 меткам):", round(result["F1"].mean(), 3))

# График: F1 для каждой метки
plt.figure(figsize=(10, 6))
plt.barh(result["Метка"], result["F1"])
plt.gca().invert_yaxis()  # лучшая метка сверху
plt.xlabel("F1")
plt.title("Качество модели по меткам (F1 на test)")
plt.tight_layout()
plt.savefig("images/f1_by_label.png")
print("\nГрафик сохранён: images/f1_by_label.png")
