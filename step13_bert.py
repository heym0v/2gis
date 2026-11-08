# BERT: готовая русская модель rubert-tiny2 (её уже обучили на большом корпусе
# текстов). Она превращает весь отзыв в вектор из 312 чисел с учётом порядка
# слов и контекста. Затем та же логистическая регрессия и сравнение
# трёх способов: TF-IDF, Word2Vec и BERT.
# Нужны библиотеки torch и transformers. При первом запуске модель
# (~120 МБ) скачивается с сайта huggingface.co.
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
labels = train.columns[3:-1]  # 19 меток (последняя колонка text_clean - не метка)

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
bert = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
bert.eval()  # режим "только использовать", модель не дообучаем


def embed(texts, batch_size=64):
    # BERT сам разбирается с формами слов, поэтому даём ему исходный текст,
    # а не очищенный text_clean
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        # max_length=128 - длинные отзывы обрезаем до 128 токенов
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        with torch.no_grad():  # без обучения, только считаем
            out = bert(**tokens)
        # вектор отзыва - вектор специального первого токена [CLS]
        cls = out.last_hidden_state[:, 0, :]
        cls = torch.nn.functional.normalize(cls)  # длина вектора = 1
        vectors.append(cls.numpy())
        if start % 10000 == 0:
            print(f"  обработано {start} из {len(texts)}")
    return np.vstack(vectors)


start = time.time()
print("Считаем векторы train...")
X_train = embed(train["text"].tolist())
print("Считаем векторы test...")
X_test = embed(test["text"].tolist())
print("Размер X_train:", X_train.shape, "| время:", round(time.time() - start), "с")

# логистическая регрессия с балансировкой, как в шагах 9 и 12
p, r, f = [], [], []
for label in labels:
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, train[label])
    pred = model.predict(X_test)
    p.append(precision_score(test[label], pred, zero_division=0))
    r.append(recall_score(test[label], pred, zero_division=0))
    f.append(f1_score(test[label], pred, zero_division=0))

print("\nBERT + логистическая регрессия (balanced), test:")
print("macro-P:", round(np.mean(p), 3), "| macro-R:", round(np.mean(r), 3),
      "| macro-F1:", round(np.mean(f), 3))

# сравнение трёх способов векторизации (модель везде одна и та же)
# числа для TF-IDF и Word2Vec - результаты шагов 9 и 12
compare = pd.DataFrame({
    "Векторизация": ["TF-IDF", "Word2Vec", "BERT (rubert-tiny2)"],
    "macro-F1": [0.438, 0.315, round(np.mean(f), 3)],
})
print("\n", compare.to_string(index=False))

plt.figure(figsize=(7, 4))
plt.bar(compare["Векторизация"], compare["macro-F1"])
for i, v in enumerate(compare["macro-F1"]):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.ylabel("macro-F1 на test")
plt.title("TF-IDF, Word2Vec и BERT\n(логистическая регрессия, balanced)")
plt.ylim(0, max(compare["macro-F1"]) + 0.1)
plt.tight_layout()
plt.savefig("images/vectorizers_compare.png")
print("\nГрафик сохранён: images/vectorizers_compare.png")
