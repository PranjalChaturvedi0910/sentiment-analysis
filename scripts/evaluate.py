# scripts/evaluate.py

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===== 1. Load pre-trained model and vectorizer (fast load) =====
model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

# ===== 2. Load only a sample for evaluation =====
print("Loading sample dataset for evaluation...")
df = pd.read_csv("../data/Tweets.csv").sample(5000, random_state=42)  # 5K rows for speed

# ===== 3. Prepare features =====
X = vectorizer.transform(df["text"])
y_true = df["airline_sentiment"]

# ===== 4. Predict =====
print("Predicting sentiments...")
y_pred = model.predict(X)

# ===== 5. Evaluation metrics =====
print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
