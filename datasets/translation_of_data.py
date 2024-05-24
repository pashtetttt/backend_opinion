import pandas as pd
import re
from mtranslate import translate
cyberbullying = pd.read_csv("/home/pashtet/projects/diploma/datasets/cyberbullying_tweets.csv")

def remove_nicknames(text):
    pattern = r'@\w+'
    return re.sub(pattern, '', text)

cyberbullying['tweet_text'] = cyberbullying['tweet_text'].apply(remove_nicknames)
cyberbullying_sh = cyberbullying.sample(frac=1).reset_index(drop=True)

df = cyberbullying_sh.head(8000)
# Создание экземпляра класса Translator


# Функция для перевода текста на русский
def translate_to_russian(text):
    try:
        translated = translate(text, 'ru', 'en')
        return translated
    except Exception as e:
        print("Translation Error:", e)
        return ""

# Применение функции translate_to_russian к каждой строке в DataFrame
df['russian_text'] = df['tweet_text'].apply(translate_to_russian)

# Сохранение результата в новый CSV файл
df.to_csv("translated_cb_dataset_1_6.csv", index=False)