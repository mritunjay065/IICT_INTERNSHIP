import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def create_report_dir():
    os.makedirs('reports/images', exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name):
    create_report_dir()
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    filename = f"reports/images/fake_news_cm_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename

def plot_roc_curves(models_dict, X_test, y_test):
    create_report_dir()
    plt.figure(figsize=(8, 6))
    
    for model_name, model in models_dict.items():
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
    plt.title('ROC Curves - Fake News Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = "reports/images/fake_news_roc_curves.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename

def plot_feature_importance(vectorizer, model, model_name, top_n=20):
    create_report_dir()
    feature_names = vectorizer.get_feature_names_out()
    
    if hasattr(model, 'coef_'):
        importances = model.coef_[0]
        title = f'Top Coef Weights (Logistic Regression)'
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title = f'Feature Importances (Random Forest)'
    else:
        return None
        
    indices = importances.argsort()[::-1]
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_weights = [importances[i] for i in top_indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_weights, y=top_features, palette='rocket')
    plt.title(title)
    plt.xlabel('Importance/Weight')
    plt.tight_layout()
    
    filename = f"reports/images/fake_news_feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename
