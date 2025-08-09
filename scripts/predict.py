import pickle
from preprocess import clean_text

# Load model & vectorizer
model = pickle.load(open("../models/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))

# Predefined neutral words
neutral_words = {"alright", "ok", "okay", "fine", "cool", "hmm"}

def predict_sentiment(text):
    # Check if text is in neutral list
    if text.lower().strip() in neutral_words:
        return "neutral"
    
    # Otherwise, use ML model
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    return model.predict(vec)[0]

if __name__ == "__main__":
    while True:
        user_input = input("Enter text (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        sentiment = predict_sentiment(user_input)
        print(f"Predicted sentiment: {sentiment}")
