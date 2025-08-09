import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
model = LinearSVC()
import pickle
from preprocess import clean_text

# Load dataset
df = pd.read_csv("../data/Tweets.csv")
df['clean_text'] = df['text'].apply(clean_text)

from sklearn.utils import resample

# Balance classes
min_class_size = df['airline_sentiment'].value_counts().min()
df = pd.concat([
    resample(df[df['airline_sentiment'] == 'positive'], replace=True, n_samples=min_class_size, random_state=42),
    resample(df[df['airline_sentiment'] == 'neutral'], replace=True, n_samples=min_class_size, random_state=42),
    resample(df[df['airline_sentiment'] == 'negative'], replace=True, n_samples=min_class_size, random_state=42)
])


# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['clean_text'])
y = df['airline_sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model & vectorizer
pickle.dump(model, open("../models/sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("../models/vectorizer.pkl", "wb"))

print("Model trained and saved!")
