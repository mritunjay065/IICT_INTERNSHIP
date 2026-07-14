import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_fake_news_models():
    """Returns initialized models for fake news classification."""
    return {
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'Logistic Regression': LogisticRegression(C=10.0, max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42, early_stopping=True)
    }

def evaluate_model(model, X_test, y_test):
    """Calculates evaluation metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    }
    return metrics, y_pred

def save_fake_news_artifacts(vectorizer, trained_models, output_dir='models'):
    """Saves vectorizer and models to pickle files."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/fake_news_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
        
    for model_name, model in trained_models.items():
        filename = f"{output_dir}/fake_news_{model_name.lower().replace(' ', '_')}_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
            
    print(f"Fake news model checkpoints successfully saved to '{output_dir}/'")
