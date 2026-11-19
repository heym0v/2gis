# облако слов для одной метки: какие слова модель считает самыми важными.
# размер слова = вес слова в модели (коэффициент логистической регрессии)
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud

# метку можно указать при запуске: python step14_wordcloud.py "Покормить детей"
label = sys.argv[1] if len(sys.argv) > 1 else "Фоткаться"

train = pd.read_csv("data/train.csv")
labels = list(train.columns[3:-1])
if label not in labels:
    sys.exit("Нет такой метки. Доступные: " + ", ".join(labels))

# как в шаге 9: TF-IDF по словам и сбалансированная логистическая регрессия
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(train["text_clean"].fillna(""))
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X, train[label])

# коэффициент слова: чем больше (больше нуля), тем сильнее слово толкает к метке
words = vectorizer.get_feature_names_out()
weights = dict(zip(words, model.coef_[0]))
weights = {w: c for w, c in weights.items() if c > 0}  # только слова "за" метку

# шрифт с кириллицей (облаку нужен файл шрифта, иначе вместо букв будут квадраты)
fonts = ["C:/Windows/Fonts/arial.ttf",                                  # Windows
         "/System/Library/Fonts/Supplemental/Arial.ttf",                # macOS
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]             # Linux
font = next((f for f in fonts if os.path.exists(f)), None)

cloud = WordCloud(width=1200, height=800, background_color="white",
                  max_words=60, font_path=font, colormap="viridis",
                  random_state=42)
cloud.generate_from_frequencies(weights)

plt.figure(figsize=(10, 6.7))
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off")
plt.title(f"Слова, на которые опирается модель: «{label.replace('Фоткаться', 'Фотографироваться')}»")
plt.tight_layout()
out = "images/wordcloud.png"
plt.savefig(out, dpi=150)
print("Топ-10 слов:", sorted(weights, key=weights.get, reverse=True)[:10])
print("Облако сохранено:", out)
