import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK downloads are available
def download_nltk_resources():
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'corpora/{res}' if res in ['stopwords', 'wordnet', 'omw-1.4'] else f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

# Initialize resources
download_nltk_resources()
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_html(text):
    """Remove HTML tags from text."""
    if not isinstance(text, str):
        return ""
    clean_re = re.compile('<.*?>')
    return re.sub(clean_re, '', text)

def preprocess_text(text):
    """
    Cleans, tokenizes, removes stopwords, and lemmatizes the text.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Clean HTML tags
    text = clean_html(text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Keep only alphabetic characters and basic spacing (removes punctuation, digits)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 4. Tokenize
    tokens = word_tokenize(text)
    
    # 5. Remove stopwords and short words, then lemmatize
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOPWORDS and len(token) > 2
    ]
    
    return " ".join(cleaned_tokens)

def extract_metadata_features(df, text_column):
    """
    Extracts structural/metadata features from raw email text.
    - url_count: number of URLs in text
    - has_urgency: flag indicating urgency keywords
    - email_length: length of the email text
    """
    # Simple URL extraction regex
    url_regex = re.compile(r'https?://\S+|www\.\S+')
    urgency_keywords = {'urgent', 'immediately', 'verify', 'suspend', 'action required', 'security alert', 'update your account'}
    
    df = df.copy()
    
    # Calculate text length
    df['email_length'] = df[text_column].apply(lambda x: len(str(x)))
    
    # Calculate URL count (before cleaning HTML/special chars)
    df['url_count'] = df[text_column].apply(lambda x: len(url_regex.findall(str(x))) if isinstance(x, str) else 0)
    
    # Check for urgency flags
    df['has_urgency'] = df[text_column].apply(
        lambda x: int(any(kw in str(x).lower() for kw in urgency_keywords)) if isinstance(x, str) else 0
    )
    
    return df
