import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_models():
    """Returns a dictionary of initialized models."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, early_stopping=True)
    }

def evaluate_model(model, X_test, y_test):
    """Calculates accuracy, precision, recall, and F1 score."""
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }
    return metrics, y_pred

def save_model_artifacts(vectorizer, trained_models, output_dir='models'):
    """Saves the vectorizer and trained models to pickle files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save vectorizer
    with open(f"{output_dir}/tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
        
    # Save each model
    for model_name, model in trained_models.items():
        filename = f"{output_dir}/{model_name.lower().replace(' ', '_')}_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
            
    print(f"Model artifacts successfully saved to '{output_dir}/' directory.")
