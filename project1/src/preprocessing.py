import re

# Standard list of English stopwords
STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now"
])

def manual_tokenize(text):
    """
    Cleans, converts to lowercase, removes punctuation/non-word characters manually, 
    and returns a list of word tokens.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation/non-word characters (replace with spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Split manually on whitespace
    tokens = [token for token in text.split(' ') if token]
    return tokens

def remove_stopwords(tokens):
    """
    Removes stopwords from a list of tokens.
    """
    return [token for token in tokens if token not in STOPWORDS]

def clean_text_pipeline(text):
    """
    Fully processes raw text to a space-separated clean token string.
    """
    tokens = manual_tokenize(text)
    cleaned_tokens = remove_stopwords(tokens)
    return " ".join(cleaned_tokens)
