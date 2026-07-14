import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.fake_news_preprocessing import clean_news_text
from src.fake_news_models import get_fake_news_models, evaluate_model, save_fake_news_artifacts
from src.fake_news_utils import plot_confusion_matrix, plot_roc_curves, plot_feature_importance

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

DATASET_URL = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
RAW_DATA_PATH = "data/raw/fake_or_real_news.csv"
CLEANED_DATA_PATH = "data/processed/cleaned_news.csv"

def download_dataset():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Downloading dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, RAW_DATA_PATH)
        print("Dataset downloaded successfully.")
    else:
        print("Raw dataset already exists.")

def main():
    download_dataset()
    
    if os.path.exists(CLEANED_DATA_PATH):
        print(f"Loading cleaned dataset from {CLEANED_DATA_PATH}...")
        df = pd.read_csv(CLEANED_DATA_PATH)
        df['cleaned_text'] = df['cleaned_text'].fillna('')
    else:
        print("Loading raw dataset...")
        df = pd.read_csv(RAW_DATA_PATH)
        df = df.dropna(subset=['text', 'label']).reset_index(drop=True)
        df['label_num'] = df['label'].map({'FAKE': 1, 'REAL': 0})
        
        print("Cleaning and preprocessing text (this may take a few minutes)...")
        tqdm.pandas()
        df['combined_text'] = df['title'] + " " + df['text']
        df['cleaned_text'] = df['combined_text'].progress_apply(clean_news_text)
        df = df[df['cleaned_text'].str.strip() != ''].reset_index(drop=True)
        
        df.to_csv(CLEANED_DATA_PATH, index=False)
        print(f"Cleaned dataset saved to {CLEANED_DATA_PATH}")
        
    X = df['cleaned_text']
    y = df['label_num']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    models = get_fake_news_models()
    trained_models = {}
    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        trained_models[name] = model
        
        print(f"Evaluating {name}...")
        metrics, y_pred = evaluate_model(model, X_test_tfidf, y_test)
        
        plot_confusion_matrix(y_test, y_pred, name)
        plot_feature_importance(vectorizer, model, name)
        
        results.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        })
        print(f"{name} - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        
    print("Plotting ROC curves...")
    plot_roc_curves(trained_models, X_test_tfidf, y_test)
    
    save_fake_news_artifacts(vectorizer, trained_models)
    
    summary_df = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print(summary_df.to_string(index=False))
    
    os.makedirs('reports', exist_ok=True)
    summary_df.to_markdown('reports/fake_news_model_performance_summary.md', index=False)

if __name__ == "__main__":
    main()
