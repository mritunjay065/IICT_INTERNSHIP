# AI-Driven Phishing Email Detection Using NLP

This folder contains a complete machine learning system built from scratch to automatically detect phishing emails using text analysis and structural metadata features.

---

## 📁 Repository Structure

*   `app/`
    *   [`app.py`](./app/app.py): Premium Streamlit dashboard for real-time email analysis.
*   `data/`
    *   `raw/phishing_emails.csv`: Raw email corpus (18,634 emails).
    *   `processed/cleaned_emails.csv`: Preprocessed, tokenized, and lemmatized text dataset.
*   `src/`
    *   [`preprocessing.py`](./src/preprocessing.py): Functions for text cleaning, stopword removal, lemmatization, and metadata extraction.
    *   [`models.py`](./src/models.py): Helper script to define models, run evaluations, and save/load models.
    *   [`utils.py`](./src/utils.py): Utility functions for plotting Confusion Matrices, ROC curves, and Feature Importances.
*   `models/`: Serialized model checkpoints and the TF-IDF vectorizer.
*   `notebooks/`
    *   [`phishing_detection.ipynb`](./notebooks/phishing_detection.ipynb): Fully documented Jupyter Notebook walkthrough containing EDA, NLP steps, and models training.
*   `reports/`
    *   [`comparative_analysis.md`](./reports/comparative_analysis.md): Detailed comparative report of the trained models.
    *   [`presentation_slides.md`](./reports/presentation_slides.md): Slide deck structured for project defense/presentation.
    *   `images/`: Exported evaluation plots (ROC curves, confusion matrices).
*   [`requirements.txt`](./requirements.txt): List of Python libraries required for this project.
*   [`download_and_train.py`](./download_and_train.py): Consolidated execution pipeline script to download data, train classifiers, and output reports.

---

## ⚡ Setup & Execution

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Full Model Pipeline
To download the dataset, preprocess text, train the models, and generate the analysis reports/plots, run:
```bash
python download_and_train.py
```

### 3. Run the Web Application
To launch the interactive Streamlit dashboard:
```bash
streamlit run app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to analyze emails.




<img width="1913" height="861" alt="image" src="https://github.com/user-attachments/assets/5110bbb8-3946-44b7-a289-199c61d3d746" />


<img width="1901" height="862" alt="image" src="https://github.com/user-attachments/assets/a5c889fe-2e0d-4704-acbf-a691a4b38b0a" />




## 📊 Comparative Performance Results

The models were evaluated on a stratified 20% test split:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Neural Network (MLP)** | **96.32%** | 94.02% | **96.79%** | **95.38%** |
| **Logistic Regression** | 96.24% | 94.48% | 96.03% | 95.25% |
| **Multinomial Naive Bayes** | 95.17% | 93.31% | 94.46% | 93.88% |
| **Random Forest** | 89.51% | **97.43%** | 75.24% | 84.91% |

---

## 📈 Visualizations
All evaluation visualizations are stored in `reports/images/`:
*   `roc_curves.png`: Combined ROC performance curves.
*   `cm_[model_name].png`: Confusion matrices detailing prediction counts.
*   `feature_importance_logistic_regression.png`: Coefficient importance indicating top phishing trigger keywords (e.g., *click*, *verify*, *update*).
