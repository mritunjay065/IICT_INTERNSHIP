import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.preprocessing import preprocess_text, extract_metadata_features
from src.models import get_models, evaluate_model, save_model_artifacts
from src.utils import plot_confusion_matrix, plot_roc_curves, plot_feature_importance

# Create directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# URL of the dataset
DATASET_URL = "https://raw.githubusercontent.com/uzmabb182/Data_622/refs/heads/main/final_project_data_622/Phishing_Email.csv"
RAW_DATA_PATH = "data/raw/phishing_emails.csv"
CLEANED_DATA_PATH = "data/processed/cleaned_emails.csv"

def download_dataset():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Downloading dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, RAW_DATA_PATH)
        print("Dataset downloaded successfully.")
    else:
        print("Raw dataset already exists.")

def main():
    download_dataset()
    
    # Load dataset
    if os.path.exists(CLEANED_DATA_PATH):
        print(f"Loading cleaned dataset from {CLEANED_DATA_PATH}...")
        df = pd.read_csv(CLEANED_DATA_PATH)
        df['cleaned_text'] = df['cleaned_text'].fillna('')
    else:
        print("Loading raw dataset...")
        try:
            df = pd.read_csv(RAW_DATA_PATH)
        except Exception as e:
            df = pd.read_csv(RAW_DATA_PATH, encoding='latin-1')
            
        print(f"Dataset shape: {df.shape}")
        print("Dataset columns:", df.columns.tolist())
        print("Class distribution:\n", df['Email Type'].value_counts())
        
        # Preprocess dataset
        df = df.dropna(subset=['Email Text', 'Email Type']).reset_index(drop=True)
        df['label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
        
        print("Extracting metadata features...")
        df = extract_metadata_features(df, 'Email Text')
        
        print("Cleaning and preprocessing email text (this may take a few minutes)...")
        tqdm.pandas()
        df['cleaned_text'] = df['Email Text'].progress_apply(preprocess_text)
        df = df[df['cleaned_text'].str.strip() != ''].reset_index(drop=True)
        
        df.to_csv(CLEANED_DATA_PATH, index=False)
        print(f"Cleaned dataset saved to {CLEANED_DATA_PATH}")
    
    # Split dataset
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Feature Extraction (TF-IDF)
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Get models
    models = get_models()
    trained_models = {}
    results = []
    
    # Train and evaluate models
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        trained_models[name] = model
        
        print(f"Evaluating {name}...")
        metrics, y_pred = evaluate_model(model, X_test_tfidf, y_test)
        
        # Plot confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, name)
        
        # Plot feature importance if supported
        plot_feature_importance(vectorizer, model, name)
        
        results.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        })
        
        print(f"{name} Metrics - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        
    # Plot combined ROC Curves
    print("Plotting ROC curves...")
    plot_roc_curves(trained_models, X_test_tfidf, y_test)
    
    # Save model artifacts
    save_model_artifacts(vectorizer, trained_models)
    
    # Display summary table
    summary_df = pd.DataFrame(results)
    print("\nModel Performance Comparison:")
    print(summary_df.to_string(index=False))
    
    # Save comparison as markdown
    os.makedirs('reports', exist_ok=True)
    summary_df.to_markdown('reports/model_performance_summary.md', index=False)

if __name__ == "__main__":
    main()
