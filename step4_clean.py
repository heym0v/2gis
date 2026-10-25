# чистим таблицу: проверяем пропуски и убираем повторы текстов.
import pandas as pd

df = pd.read_csv("data/reviews.csv")
print("Строк до чистки:", len(df))

# 1. пропуски: сколько пустых значений в каждой колонке
print("Пропусков всего:", df.isna().sum().sum())

# 2. повторы: один и тот же текст у нескольких заведений
repeated = df.duplicated("text", keep=False)
print("Строк с повторяющимся текстом:", repeated.sum())
print("Разных таких текстов:", df[repeated]["text"].nunique())

# оставляем только первую копию каждого текста
df = df.drop_duplicates("text", keep="first")
print("Строк после чистки:", len(df))


df.to_csv("data/reviews_clean.csv", index=False)
print("Сохранено в data/reviews_clean.csv")
