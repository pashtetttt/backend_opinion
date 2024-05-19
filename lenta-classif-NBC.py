import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Загрузка датасета
data = pd.read_csv("/home/pashtet/projects/diploma/datasets/lenta.csv")
data = data.dropna()
# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['topic'], test_size=0.2, random_state=42)

# Преобразование текста в матрицу признаков
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Обучение наивного байесовского классификатора
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Предсказание на тестовом наборе
y_pred = clf.predict(X_test_counts)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Вывод отчета о классификации
print(classification_report(y_test, y_pred))
