from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd


df = pd.read_csv("datasets/lenta.csv")
df.dropna()
# Предобработка текста и подготовка данных
X = df['text']  # 'text' - это столбец с текстом
y = df['tags']  # 'tags' - это столбец с метками

# Разделение на обучающий и тестовый наборы данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание Bag of Words
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

# Преобразование в TF-IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Обучение SVM модели
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Оценка производительности модели
accuracy = svm_model.score(X_test_tfidf, y_test)
print("Accuracy:", accuracy)
