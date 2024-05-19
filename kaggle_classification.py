import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# Загрузка датасета
data = pd.read_csv("text_series_full.csv")  # Замените "your_dataset.csv" на путь к вашему датасету
data.dropna()
# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.2, random_state=42)

# Создание конвейера обработки данных и обучения модели 
model = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Векторизация текста
    ('clf', SVC(kernel='linear'))  # Классификатор (можно выбрать другой, например, RandomForestClassifier)
])

# Обучение модели
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Оценка модели на тестовом наборе

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Выведите результаты
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
# Сохранение модели в файл
# joblib.dump(model, "text_classification_model.pkl")  # Сохраняем модель в файл "text_classification_model.pkl"
cm = confusion_matrix(y_test, y_pred)

# Создайте теплокарту для визуализации матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()