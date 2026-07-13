# Comparative Analysis Report: Phishing Email Detection using NLP

## Executive Summary
This report analyzes the performance of four machine learning models applied to the task of automatically detecting phishing emails from raw email text: **Logistic Regression**, **Multinomial Naive Bayes**, **Random Forest**, and a **Multi-Layer Perceptron (Neural Network)**. The dataset consists of over 18,600 emails with balanced representation of safe and phishing instances. Models were trained on TF-IDF features extracted from preprocessed email text. 

The **Neural Network** and **Logistic Regression** emerged as the top-performing models, both achieving an accuracy of **~96.3%** and an F1-Score of **~95.3%**.

---

## Dataset & Preprocessing
The dataset used is the combined **Enron and Nazario corpus** (18,634 emails after cleaning nulls).
1. **Linguistic Preprocessing:**
   * HTML tag removal (regex-based).
   * Casing standardization (lowercasing).
   * Punctuation, numbers, and special character removal.
   * Tokenization and NLTK stopword filtering.
   * WordNet Lemmatization.
2. **Feature Extraction:**
   * Text converted to numerical vectors using **TF-IDF vectorization** (max 5,000 features).
   * 80% train / 20% test split.

---

## Model Evaluation Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Neural Network (MLP)** | **96.32%** | 94.02% | **96.79%** | **95.38%** |
| **Logistic Regression** | 96.24% | 94.48% | 96.03% | 95.25% |
| **Multinomial Naive Bayes** | 95.17% | 93.31% | 94.46% | 93.88% |
| **Random Forest** | 89.51% | **97.43%** | 75.24% | 84.91% |

---

## Performance Interpretation & Analysis

### 1. Neural Network (MLP) - Best Overall F1 & Recall
* **Accuracy:** 96.32% | **F1-Score:** 95.38% | **Recall:** 96.79%
* **Interpretation:** The MLP model excels at capturing non-linear interactions between keywords. It has the highest recall (96.79%), meaning it minimizes false negatives (phishing emails slipping into the inbox), which is the most critical requirement in cybersecurity.

### 2. Logistic Regression - Best Speed-to-Accuracy Balance
* **Accuracy:** 96.24% | **F1-Score:** 95.25%
* **Interpretation:** Performs almost identically to the Neural Network, but executes in milliseconds. Logistic Regression works extremely well in high-dimensional sparse TF-IDF spaces because of its linear classification boundary, making it the most practical model for production.

### 3. Multinomial Naive Bayes - Fast and Strong Baseline
* **Accuracy:** 95.17% | **F1-Score:** 93.88%
* **Interpretation:** Naive Bayes assumes independent features, which fits word frequencies well. It achieves strong results with virtually zero training overhead.

### 4. Random Forest - High Precision, Lower Recall
* **Accuracy:** 89.51% | **Precision:** 97.43% | **Recall:** 75.24%
* **Interpretation:** Random Forest achieves the highest precision (97.43%), meaning when it flags an email as phishing, it is almost certainly correct (very few false positives). However, it misses about 25% of actual phishing attempts (low recall), making it less ideal as a standalone cybersecurity defense.

---

## Key Visualizations
* **Confusion Matrices:** Shows exact counts of True Positives, False Positives, True Negatives, and False Negatives for each model.
* **ROC Curves:** Plotted combined ROC curves demonstrate that Neural Network and Logistic Regression have the highest Area Under the Curve (AUC ~0.99), proving their superior discriminative power.
* **Feature Importance:** Logistic Regression coefficient analysis shows that keywords like *click*, *account*, *link*, *update*, *please*, and *verify* carry the highest weights for predicting phishing.

---

## Recommendations
For an enterprise implementation:
1. **Primary Choice:** **Logistic Regression** is recommended due to its high efficiency, low memory footprint, and near-perfect performance (~96.2% accuracy).
2. **Hybrid Approach:** Deploy Logistic Regression as a fast, first-pass filter, and route borderline cases to the **Neural Network (MLP)** for deeper pattern verification.
