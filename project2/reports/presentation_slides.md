# Presentation Slides: AI-Driven Phishing Email Detection Using NLP

---

## Slide 1: Project Overview
### AI-Driven Phishing Email Detection Using NLP
* **Objective:** Design and implement a machine learning system that automatically classifies emails as Safe or Phishing.
* **Core Technology:** Natural Language Processing (NLP) combined with Classification Classifiers.
* **Academic Context:** Course Project - Indian Institute of Computing and Technology (IICT).

---

## Slide 2: Methodology & Workflow
### Modular ML Pipeline Structure:
1. **Data Collection:** Sourced 18,634 raw emails (Enron + Nazario).
2. **Text Cleaning:** HTML removal, lowercase conversion, punctuation filtering.
3. **NLP Processing:** Tokenization, stopword removal, and Lemmatization.
4. **Feature Engineering:** TF-IDF representation (5,000 max features) + metadata analysis (URLs, urgency flags).
5. **Model Development:** Trained 4 classifiers (Logistic Regression, Naive Bayes, Random Forest, Neural Network).

---

## Slide 3: Model Evaluation Results
### Performance Summary (Test Set):

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Neural Network (MLP)** | **96.32%** | 94.02% | **96.79%** | **95.38%** |
| **Logistic Regression** | 96.24% | 94.48% | 96.03% | 95.25% |
| **Multinomial Naive Bayes** | 95.17% | 93.31% | 94.46% | 93.88% |
| **Random Forest** | 89.51% | **97.43%** | 75.24% | 84.91% |

---

## Slide 4: Key Insights & Visualizations
### Summary of Findings:
* **The "click/verify" pattern:** Feature importance coefficients highlight that urgency words (e.g. *verify*, *update*, *suspend*, *link*) are the primary signals of phishing attempts.
* **Tradeoffs:** 
  * **Neural Network (MLP)** provides the lowest false-negative rate (Recall: 96.79%).
  * **Random Forest** provides the lowest false-positive rate (Precision: 97.43%), but misses too many threats.
  * **Logistic Regression** achieves optimal speed/accuracy trade-off.

---

## Slide 5: Web Deployment Demo
### Streamlit Interactive Application:
* **Live Analysis:** Paste any raw email and get immediate results.
* **Confidence Level:** Displays percentage probabilities of classification.
* **Key Indicators:** Highlights detected URLs, word count, and urgency indicators.
* **Cross-Model Check:** Runs predictions across all 4 trained models simultaneously for comparisons.

---

## Slide 6: Recommendations & Next Steps
### Project Recommendations:
1. **Model Selection:** Deploy the **Logistic Regression** classifier for real-time edge processing due to fast prediction speeds and high F1-score (~95.3%).
2. **Security Integration:** Combine NLP features with email header analysis (SPF, DKIM validation) to improve accuracy.
3. **Future Scope:** Fine-tune pre-trained Transformer models (like DistilBERT) for context-aware phishing detection.
