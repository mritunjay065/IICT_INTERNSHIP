import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def download_nltk_resources():
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'corpora/{res}' if res in ['stopwords', 'wordnet', 'omw-1.4'] else f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

download_nltk_resources()
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_news_text(text):
    """
    Cleans, tokenizes, removes stopwords, and lemmatizes news text.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOPWORDS and len(token) > 2
    ]
    return " ".join(cleaned_tokens)
