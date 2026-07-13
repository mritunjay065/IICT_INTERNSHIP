import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def create_report_dir():
    os.makedirs('reports/images', exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix heatmap."""
    create_report_dir()
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Safe', 'Phishing'],
                yticklabels=['Safe', 'Phishing'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    filename = f"reports/images/cm_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename

def plot_roc_curves(models_dict, X_test, y_test):
    """Plot combined ROC curve for all models and save it."""
    create_report_dir()
    plt.figure(figsize=(8, 6))
    
    for model_name, model in models_dict.items():
        # Get probability estimates
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
            
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = "reports/images/roc_curves.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename

def plot_feature_importance(vectorizer, model, model_name, top_n=20):
    """Plot and save feature importance for models that support it (e.g. Random Forest, Logistic Regression)."""
    create_report_dir()
    feature_names = vectorizer.get_feature_names_out()
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title = f'Feature Importances - {model_name}'
    elif hasattr(model, 'coef_'):
        # For binary Logistic Regression, coef_ is shape (1, n_features)
        importances = model.coef_[0]
        title = f'Top Coefficient Weights (Logistic Regression)'
    else:
        return None
        
    # Sort features
    indices = importances.argsort()[::-1]
    
    # We want top N positive weights or top N absolute weights
    plt.figure(figsize=(10, 6))
    
    # Take top N
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_weights = [importances[i] for i in top_indices]
    
    sns.barplot(x=top_weights, y=top_features, palette='viridis')
    plt.title(title)
    plt.xlabel('Importance/Weight')
    plt.tight_layout()
    
    filename = f"reports/images/feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename
