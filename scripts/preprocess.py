import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean tweet text: lowercase, remove URLs, punctuation, stopwords.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Keep only letters
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)
