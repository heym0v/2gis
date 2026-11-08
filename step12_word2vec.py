# Word2Vec: каждое слово превращаем в вектор из 100 чисел (его учит маленькая
# нейросеть), отзыв = среднее векторов его слов. Затем та же логистическая
# регрессия, что и раньше, и сравнение с TF-IDF.
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
labels = train.columns[3:-1]  # 19 меток (последняя колонка text_clean - не метка)

# Word2Vec учится на списках слов, поэтому делим очищенный текст по пробелам
train_words = train["text_clean"].str.split().tolist()
test_words = test["text_clean"].str.split().tolist()

# обучаем Word2Vec только на train (в test не подглядываем)
# vector_size=100 - длина вектора слова, window=5 - сколько соседних слов
# учитывать, min_count=3 - слова реже 3 раз пропускаем, seed - для повторяемости
w2v = Word2Vec(train_words, vector_size=100, window=5, min_count=3,
               workers=1, seed=42, epochs=10)
print("Слов в словаре Word2Vec:", len(w2v.wv))

# проверка: какие слова модель считает похожими по смыслу
for word in ["свидание", "ребёнок", "ноутбук"]:
    similar = [w for w, _ in w2v.wv.most_similar(word, topn=5)]
    print(f"Похожие на «{word}»: {', '.join(similar)}")


def review_vector(words):
    # вектор отзыва = среднее векторов его слов (незнакомые слова пропускаем)
    vectors = [w2v.wv[w] for w in words if w in w2v.wv]
    if len(vectors) == 0:
        return np.zeros(100)
    return np.mean(vectors, axis=0)


X_train = np.array([review_vector(words) for words in train_words])
X_test = np.array([review_vector(words) for words in test_words])
print("Размер X_train:", X_train.shape)

# логистическая регрессия с балансировкой, как в шаге 9
p, r, f = [], [], []
for label in labels:
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, train[label])
    pred = model.predict(X_test)
    p.append(precision_score(test[label], pred, zero_division=0))
    r.append(recall_score(test[label], pred, zero_division=0))
    f.append(f1_score(test[label], pred, zero_division=0))

print("\nWord2Vec + логистическая регрессия (balanced), test:")
print("macro-P:", round(np.mean(p), 3), "| macro-R:", round(np.mean(r), 3),
      "| macro-F1:", round(np.mean(f), 3))
print("Для сравнения TF-IDF + та же модель (шаг 9): macro-F1 = 0.438")
