import re
import nltk
from nltk.corpus import stopwords


def ensure_nltk_resources():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word and word not in stop_words]
    return ' '.join(tokens)
