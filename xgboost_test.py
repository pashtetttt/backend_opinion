import xgboost as xgb
import pickle

model = xgb.Booster()
model.load_model("/home/pashtet/projects/diploma/xgboost_model_10_iterations.model")
text = ["хуй"]
with open('count_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

X_new_counts = vectorizer.transform(text)

# Convert to DMatrix
dnew = xgb.DMatrix(X_new_counts)

# Make predictions
predictions = model.predict(dnew)

# Decode the predictions to get the original labels
predicted_labels = label_encoder.inverse_transform(predictions.astype(int))

# Print the predicted labels
print(predicted_labels)