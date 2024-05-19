import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Загрузка предобученной модели BERT и токенизатора
model_name = 'bert-base-multilingual-cased'  # Предобученная модель BERT для многоязычного текста
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)  # Предполагается два класса

# Пример входных данных для классификации текста
text = "Это пример текста для классификации на русском языке."

# Токенизация текста и подготовка к вводу в модель
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Классификация текста с помощью модели BERT
with torch.no_grad():
    outputs = model(**inputs)

# Получение вероятностей принадлежности к классам
probs = softmax(outputs.logits, dim=1)

# Вывод результатов
print("Вероятности классов:", probs)
