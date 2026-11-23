# F1 и каппа Коэна итоговой модели (шаг 13: биграммы + свой порог) по каждой метке
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, cohen_kappa_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
labels = train.columns[3:-1]  # 19 меток

# всё как в шаге 13
fit = train.sample(frac=0.8, random_state=42)
val = train.drop(fit.index)
vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
Xf = vec.fit_transform(fit["text_clean"].fillna(""))
Xv = vec.transform(val["text_clean"].fillna(""))
Xt = vec.transform(test["text_clean"].fillna(""))

rows = []
for label in labels:
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(Xf, fit[label])
    proba_val = model.predict_proba(Xv)[:, 1]
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(val[label], proba_val >= t)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    pred = model.predict_proba(Xt)[:, 1] >= best_t
    rows.append({"Метка": label,
                 "Доля в test, %": round(100 * test[label].mean(), 1),
                 "F1": f1_score(test[label], pred),
                 "каппа": cohen_kappa_score(test[label], pred)})

result = pd.DataFrame(rows).sort_values("каппа", ascending=False)
result["Метка"] = result["Метка"].replace({"Фоткаться": "Фотографироваться"})
print(result.round(2).to_string(index=False))
print("\nmacro-F1:", round(result["F1"].mean(), 3), " macro-каппа:", round(result["каппа"].mean(), 3))
result.round(3).to_csv("images/kappa_by_label.csv", index=False)

# график: F1 и каппа по меткам
y = np.arange(len(result))
plt.figure(figsize=(10, 8))
plt.barh(y - 0.2, result["F1"], height=0.4, label="F1")
plt.barh(y + 0.2, result["каппа"], height=0.4, label="каппа Коэна")
plt.yticks(y, result["Метка"])
plt.gca().invert_yaxis()
plt.xlabel("значение на test")
plt.title("F1 и каппа Коэна по меткам (итоговая модель)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("images/kappa_by_label.png")
print("График сохранён: images/kappa_by_label.png")
