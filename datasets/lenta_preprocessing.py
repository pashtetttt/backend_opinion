import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Загрузка вашего датасета
df = pd.read_csv("lenta.csv")

# Приведение текста к нижнему регистру
df['text'] = df['text'].str.lower()

# Удаление пунктуации
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['text'] = df['text'].apply(remove_punctuation)

# Токенизация текста
def tokenize_text(text):
    return word_tokenize(text)

df['tokens'] = df['text'].apply(tokenize_text)

# Удаление стоп-слов
stop_words = set(stopwords.words('english')) 

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

df['tokens'] = df['tokens'].apply(remove_stopwords)

# Лемматизация слов
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

df['tokens'] = df['tokens'].apply(lemmatize_tokens)

# Сохранение предварительно обработанных данных в новый файл
df.to_csv("preprocessed_dataset.csv", index=False)
