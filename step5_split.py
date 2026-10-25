# делим чистую таблицу на train (80%) и test (20%).
import pandas as pd

df = pd.read_csv("data/reviews_clean.csv")
labels = df.columns[3:]

# случайно берём 80% строк в train. random_state=42 - чтобы при каждом запуске
# получалось одно и то же разбиение.
train = df.sample(frac=0.8, random_state=42)
# всё, что не попало в train, идёт в test
test = df.drop(train.index)

print("Всего:", len(df), "| train:", len(train), "| test:", len(test))

# проверка: ни один текст не должен быть сразу и в train, и в test
common = set(train["text"]) & set(test["text"])
print("Общих текстов в train и test:", len(common))

# проверка: сколько каждой метки попало в train и test
print("\nМетка: train / test")
for label in labels:
    print(f"{label}: {train[label].sum()} / {test[label].sum()}")

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
print("\nСохранено: data/train.csv и data/test.csv")
