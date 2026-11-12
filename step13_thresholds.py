# улучшаем модель из шага 9 двумя способами:
# 1) биграммы: TF-IDF учитывает не только слова, но и пары слов ("не вкусно")
# 2) свой порог для каждой метки вместо общего 0.5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
labels = train.columns[3:-1]  # 19 меток (text_clean в конце - не метка)

# пороги нельзя подбирать на test (это как подглядывать в ответы),
# поэтому откладываем 20% train как проверочную часть (validation)
fit = train.sample(frac=0.8, random_state=42)
val = train.drop(fit.index)
print("Обучение:", len(fit), " проверка:", len(val), " test:", len(test))

# вариант шага 9: только отдельные слова, 10 000 признаков
words_vec = TfidfVectorizer(max_features=10000)
Xf_words = words_vec.fit_transform(fit["text_clean"].fillna(""))
Xt_words = words_vec.transform(test["text_clean"].fillna(""))

# новый вариант: слова + пары слов (ngram_range=(1, 2)), 30 000 признаков
bi_vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
Xf = bi_vec.fit_transform(fit["text_clean"].fillna(""))
Xv = bi_vec.transform(val["text_clean"].fillna(""))
Xt = bi_vec.transform(test["text_clean"].fillna(""))

thresholds_to_try = np.arange(0.05, 0.95, 0.05)  # 0.05, 0.10, ..., 0.90
rows = []
for label in labels:
    # 1) как в шаге 9: слова, class_weight="balanced", порог 0.5
    m_words = LogisticRegression(max_iter=1000, class_weight="balanced")
    m_words.fit(Xf_words, fit[label])
    f1_words = f1_score(test[label], m_words.predict(Xt_words))

    # 2) слова + биграммы, порог 0.5
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(Xf, fit[label])
    proba_val = model.predict_proba(Xv)[:, 1]   # вероятность метки на validation
    proba_test = model.predict_proba(Xt)[:, 1]  # вероятность метки на test
    f1_bi = f1_score(test[label], proba_test >= 0.5)

    # 3) подбираем порог на validation: перебираем и берём тот, где F1 больше
    best_t, best_f1 = 0.5, 0
    for t in thresholds_to_try:
        f1 = f1_score(val[label], proba_val >= t)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    # и только потом один раз проверяем этот порог на test
    f1_thr = f1_score(test[label], proba_test >= best_t)

    rows.append({"Метка": label, "Порог": round(best_t, 2),
                 "F1 шаг 9": f1_words, "F1 +биграммы": f1_bi,
                 "F1 +биграммы +порог": f1_thr})

result = pd.DataFrame(rows).sort_values("F1 +биграммы +порог", ascending=False)
print(result.round(2).to_string(index=False))
print("\nmacro-F1 шаг 9 (слова, порог 0.5):", round(result["F1 шаг 9"].mean(), 3))
print("macro-F1 + биграммы:              ", round(result["F1 +биграммы"].mean(), 3))
print("macro-F1 + биграммы + свои пороги:", round(result["F1 +биграммы +порог"].mean(), 3))

# график: F1 по меткам для трёх вариантов
y = np.arange(len(result))
plt.figure(figsize=(10, 8))
plt.barh(y - 0.27, result["F1 шаг 9"], height=0.27, label="шаг 9: слова, порог 0.5")
plt.barh(y, result["F1 +биграммы"], height=0.27, label="+ биграммы")
plt.barh(y + 0.27, result["F1 +биграммы +порог"], height=0.27, label="+ биграммы + свой порог")
plt.yticks(y, result["Метка"])
plt.gca().invert_yaxis()
plt.xlabel("F1 на test")
plt.title("F1 по меткам: биграммы и подбор порога")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("images/f1_thresholds.png")
print("\nГрафик сохранён: images/f1_thresholds.png")
