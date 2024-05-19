import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Загрузка модели VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Пример текста для анализа
text = "I really liked this movie!"

# Анализ тональности текста
scores = sia.polarity_scores(text)

# Вывод результатов
print("Scores:", scores)