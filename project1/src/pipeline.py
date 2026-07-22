import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from preprocessing import clean_text_pipeline
from feature_extraction import TfidfVectorizerScratch
from models_scratch import (
    KNNClassifierScratch,
    LogisticRegressionScratch,
    RandomForestScratch,
    NeuralNetworkScratch
)
from models_sklearn import get_sklearn_models
from evaluation import (
    accuracy_score_scratch,
    precision_score_scratch,
    recall_score_scratch,
    f1_score_scratch,
    confusion_matrix_scratch,
    classification_report_scratch
)
from eda import run_eda

def run_pipeline(csv_path="GlobalFakeNews_Research2026_v1.csv", output_dir="results"):
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Run EDA
    run_eda(csv_path, plot_dir)
    
    # 2. Load Data
    print("\nLoading dataset...")
    df = pd.read_csv(csv_path)
    
    # We will use the full text
    X_raw = df['full_text'].values
    y = df['label'].values
    
    # 3. Clean Text using Custom Preprocessing
    print("Preprocessing text manually (lowercase, punctuation, stopword removal)...")
    start_time = time.time()
    X_clean = np.array([clean_text_pipeline(text) for text in X_raw])
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")
    
    # 4. Feature Extraction using Custom TF-IDF
    # We restrict max_features to 1000 to ensure the custom DecisionTree/RandomForest runs fast while maintaining accuracy.
    print("Extracting TF-IDF features using custom TfidfVectorizerScratch...")
    start_time = time.time()
    vectorizer = TfidfVectorizerScratch(max_features=1000)
    X_vec = vectorizer.fit_transform(X_clean)
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds. Matrix shape: {X_vec.shape}")
    
    # 5. Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    # 6. Initialize models
    scratch_models = {
        "KNN (Scratch)": KNNClassifierScratch(n_neighbors=5, metric='cosine'),
        "LogReg (Scratch)": LogisticRegressionScratch(lr=0.5, n_iters=1000),
        "RandomForest (Scratch)": RandomForestScratch(n_estimators=10, max_depth=6),
        "NeuralNet (Scratch)": NeuralNetworkScratch(input_size=X_train.shape[1], hidden_size=32, lr=0.1, epochs=150, batch_size=32)
    }
    
    sklearn_models = get_sklearn_models()
    
    results = {}
    
    # --- Train & Evaluate Custom Models ---
    print("\nTraining and evaluating models built from scratch...")
    for name, model in scratch_models.items():
        print(f"Training {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        print(f"Predicting with {name}...")
        t0 = time.time()
        preds = model.predict(X_test)
        pred_time = time.time() - t0
        
        acc = accuracy_score_scratch(y_test, preds)
        prec = precision_score_scratch(y_test, preds)
        rec = recall_score_scratch(y_test, preds)
        f1 = f1_score_scratch(y_test, preds)
        cm = confusion_matrix_scratch(y_test, preds)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ConfusionMatrix": cm.tolist(),
            "TrainTime": train_time,
            "PredTime": pred_time
        }
        print(f"{name} Results:")
        print(classification_report_scratch(y_test, preds))
        
        # Save training loss for NN
        if name == "NeuralNet (Scratch)":
            plt.figure(figsize=(8, 5))
            plt.plot(model.loss_history, color='royalblue', lw=2)
            plt.title('Neural Network (Scratch) Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Binary Cross-Entropy Loss')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "nn_loss_curve.png"), dpi=300)
            plt.close()
            
    # --- Train & Evaluate Sklearn Baseline Models ---
    print("\nTraining and evaluating scikit-learn models (benchmarks)...")
    for name, model in sklearn_models.items():
        name_with_suffix = f"{name} (Sklearn)"
        print(f"Training {name_with_suffix}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        print(f"Predicting with {name_with_suffix}...")
        t0 = time.time()
        preds = model.predict(X_test)
        pred_time = time.time() - t0
        
        acc = accuracy_score_scratch(y_test, preds)
        prec = precision_score_scratch(y_test, preds)
        rec = recall_score_scratch(y_test, preds)
        f1 = f1_score_scratch(y_test, preds)
        cm = confusion_matrix_scratch(y_test, preds)
        
        results[name_with_suffix] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ConfusionMatrix": cm.tolist(),
            "TrainTime": train_time,
            "PredTime": pred_time
        }
        print(f"{name_with_suffix} Results:")
        print(classification_report_scratch(y_test, preds))

    # Save results to json
    with open(os.path.join(output_dir, "model_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # 7. Plots and Visualizations of Comparison
    print("\nGenerating final comparison charts...")
    df_metrics = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    
    # Accuracies Bar Chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_metrics, x='Model', y='Accuracy', palette='viridis')
    plt.title('Accuracy Comparison: Scratch vs Sklearn')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=30, ha='right')
    plt.ylim(0.7, 1.05)
    for index, row in df_metrics.iterrows():
        plt.text(index, row['Accuracy'] + 0.01, f"{row['Accuracy']:.4f}", color='black', ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "accuracy_comparison.png"), dpi=300)
    plt.close()
    
    # Save confusion matrices plots
    for name in results.keys():
        cm = np.array(results[name]["ConfusionMatrix"])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Real (0)", "Fake (1)"],
                    yticklabels=["Real (0)", "Fake (1)"])
        plt.title(f"Confusion Matrix: {name}")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{name.replace(' ', '_').lower()}.png"), dpi=300)
        plt.close()
        
    print(f"\nAll models trained and evaluated. Output reports and plots saved to '{output_dir}/'")

if __name__ == "__main__":
    run_pipeline()
