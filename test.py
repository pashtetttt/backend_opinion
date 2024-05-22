import joblib

loaded_model = joblib.load("text_classification_model.pkl")
pr_class = loaded_model.predict(["Государство государство"])
print(pr_class[0])