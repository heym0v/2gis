# Шаг 1. Загружаем JSON и смотрим, что внутри.
import json

# Открываем файл и читаем его целиком в переменную data
with open("data/task_2_vibes.json", encoding="utf-8") as f:
    data = json.load(f)

# data - это список. Один элемент списка = одно заведение (кафе, ресторан)
print("Всего заведений:", len(data))

# Смотрим на первое заведение
first = data[0]
print("Поля заведения:", list(first.keys()))
print("Название:", first["caption"])
print("Рейтинг:", first["rating"])

# Внутри first["nlp"]["aspect"] лежат метки.
# Это словарь: метка -> {текст отзыва: [фразы, почему поставили метку]}
aspects = first["nlp"]["aspect"]
print("Метки у этого заведения:", list(aspects.keys()))

# Показываем один отзыв для метки "Свидание"
for review, phrases in aspects["Свидание"].items():
    print("Отзыв:", review)
    print("Почему метка:", phrases)

# Считаем, сколько раз встречается каждая метка во всём файле
counts = {}
for place in data:
    for label, reviews in place["nlp"]["aspect"].items():
        counts[label] = counts.get(label, 0) + len(reviews)

print("Всего разных меток:", len(counts))
for label, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(label, n)
