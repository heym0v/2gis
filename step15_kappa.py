# итоговая таблица сравнения моделей: кроме точности, полноты и F1
# считаем каппу Коэна и разброс (минимум - максимум) по 19 меткам.
# Каппа показывает, насколько ответы модели совпадают с разметкой
# сверх случайного совпадения: 0 - как наугад, 1 - полное совпадение.
# При дисбалансе классов она строже, чем доля верных ответов.
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
labels = train.columns[3:-1]  # 19 меток (text_clean в конце - не метка)
train_text = train["text_clean"].fillna("")
test_text = test["text_clean"].fillna("")


def scores(name, predictions):
    # predictions: словарь {метка: ответы модели 0/1 на test}
    p, r, f, k = [], [], [], []
    for label in labels:
        y, pred = test[label], predictions[label]
        p.append(precision_score(y, pred, zero_division=0))
        r.append(recall_score(y, pred, zero_division=0))
        f.append(f1_score(y, pred, zero_division=0))
        k.append(cohen_kappa_score(y, pred))
    print(name, "- готово")
    return {"Модель": name,
            "P": np.mean(p), "R": np.mean(r),
            "F1": np.mean(f), "F1 мин": min(f), "F1 макс": max(f),
            "каппа": np.mean(k), "каппа мин": min(k), "каппа макс": max(k)}


rows = []

# 1) TF-IDF по словам (как в шагах 6-11) и пять моделей
vec = TfidfVectorizer(max_features=10000)
X_train = vec.fit_transform(train_text)
X_test = vec.transform(test_text)
models = {
    "Наивный Байес (TF-IDF)": lambda: MultinomialNB(),
    "Лог. регрессия (TF-IDF)": lambda: LogisticRegression(max_iter=1000),
    "Линейный SVM (TF-IDF)": lambda: LinearSVC(),
    "Линейный SVM, balanced (TF-IDF)": lambda: LinearSVC(class_weight="balanced"),
    "Лог. регрессия, balanced (TF-IDF)": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
}
for name, make_model in models.items():
    preds = {}
    for label in labels:
        model = make_model()
        model.fit(X_train, train[label])
        preds[label] = model.predict(X_test)
    rows.append(scores(name, preds))

# 2) Word2Vec (как в шаге 12): отзыв = среднее векторов слов
train_words = train_text.str.split().tolist()
test_words = test_text.str.split().tolist()
w2v = Word2Vec(train_words, vector_size=100, window=5, min_count=3,
               workers=1, seed=42, epochs=10)


def review_vector(words):
    vectors = [w2v.wv[w] for w in words if w in w2v.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)


W_train = np.array([review_vector(w) for w in train_words])
W_test = np.array([review_vector(w) for w in test_words])
preds = {}
for label in labels:
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(W_train, train[label])
    preds[label] = model.predict(W_test)
rows.append(scores("Лог. регрессия, balanced (Word2Vec)", preds))

# 3) итоговая модель (как в шаге 13): биграммы + свой порог для каждой метки
fit = train.sample(frac=0.8, random_state=42)
val = train.drop(fit.index)
bi_vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
Xf = bi_vec.fit_transform(fit["text_clean"].fillna(""))
Xv = bi_vec.transform(val["text_clean"].fillna(""))
Xt = bi_vec.transform(test_text)
preds = {}
for label in labels:
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(Xf, fit[label])
    proba_val = model.predict_proba(Xv)[:, 1]
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(val[label], proba_val >= t)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    preds[label] = (model.predict_proba(Xt)[:, 1] >= best_t).astype(int)
rows.append(scores("Лог. регрессия, balanced (TF-IDF + биграммы, свои пороги)", preds))

result = pd.DataFrame(rows).sort_values("F1")
print("\nСравнение моделей на test (среднее по 19 меткам, мин и макс по меткам):")
print(result.round(2).to_string(index=False))
result.round(3).to_csv("images/models_kappa.csv", index=False)
print("\nТаблица сохранена: images/models_kappa.csv")
