import json
import os

os.makedirs('notebooks', exist_ok=True)

cells = []

# Title & Intro
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# AI-Driven Phishing Email Detection Using NLP\n",
        "**Institution:** Indian Institute of Computing and Technology (IICT)\n",
        "\n",
        "### Project Objectives:\n",
        "1. **Data Preprocessing:** Clean email text by removing HTML tags, punctuation, and stopwords, and apply lemmatization.\n",
        "2. **Feature Engineering:** Extract linguistic features using TF-IDF and structural metadata (e.g., URL count, urgency keywords).\n",
        "3. **Model Development:** Train and evaluate multiple classifiers (Logistic Regression, Random Forest, Naive Bayes, Neural Network).\n",
        "4. **Comparative Evaluation:** Compare performance using Accuracy, Precision, Recall, and F1-score to identify the best model."
    ]
})

# Setup
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 1: Import Dependencies and Download NLTK Data"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import re\n",
        "import pickle\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "from nltk.tokenize import word_tokenize\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.neural_network import MLPClassifier\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc\n",
        "\n",
        "# Download NLTK resources\n",
        "nltk.download('stopwords', quiet=True)\n",
        "nltk.download('wordnet', quiet=True)\n",
        "nltk.download('omw-1.4', quiet=True)\n",
        "nltk.download('punkt', quiet=True)\n",
        "nltk.download('punkt_tab', quiet=True)\n",
        "print(\"Libraries imported and NLTK data loaded successfully!\")"
    ]
})

# Load Data
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 2: Load the Dataset\n",
        "We load the downloaded raw phishing email dataset."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "data_path = \"../data/raw/phishing_emails.csv\"\n",
        "if not os.path.exists(data_path):\n",
        "    # Fallback to local project directory path\n",
        "    data_path = \"data/raw/phishing_emails.csv\"\n",
        "\n",
        "df = pd.read_csv(data_path)\n",
        "print(f\"Dataset Shape: {df.shape}\")\n",
        "print(\"\\nFirst 5 rows:\")\n",
        "display(df.head())\n",
        "print(\"\\nClass Distribution:\")\n",
        "print(df['Email Type'].value_counts())"
    ]
})

# EDA / Visualization of classes
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Exploratory Data Analysis (EDA)\n",
        "Let's visualize the ratio of Legitimate (Safe) vs. Phishing emails."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "plt.figure(figsize=(6, 4))\n",
        "sns.countplot(x='Email Type', data=df, palette='Set2')\n",
        "plt.title('Distribution of Email Classes')\n",
        "plt.xlabel('Email Classification')\n",
        "plt.ylabel('Count')\n",
        "plt.grid(axis='y', alpha=0.3)\n",
        "plt.show()"
    ]
})

# Preprocessing
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 3: Data Cleaning and Text Preprocessing\n",
        "To feed clean textual features into our models, we:\n",
        "1. Remove HTML tags.\n",
        "2. Keep only alphabetic characters (remove punctuation, numbers, special signs).\n",
        "3. Standardize casing (lowercase).\n",
        "4. Tokenize the text into words.\n",
        "5. Remove English stopwords and short tokens.\n",
        "6. Lemmatize words to their dictionary forms."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "stop_words = set(stopwords.words('english'))\n",
        "lemmatizer = WordNetLemmatizer()\n",
        "\n",
        "def clean_email_text(text):\n",
        "    if not isinstance(text, str):\n",
        "        return \"\"\n",
        "    # Remove HTML tags\n",
        "    text = re.sub(r'<[^>]+>', '', text)\n",
        "    # Convert to lowercase\n",
        "    text = text.lower()\n",
        "    # Keep only letters\n",
        "    text = re.sub(r'[^a-z\\s]', '', text)\n",
        "    # Tokenize\n",
        "    tokens = word_tokenize(text)\n",
        "    # Filter stopwords and lemmatize\n",
        "    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]\n",
        "    return \" \".join(cleaned)\n",
        "\n",
        "# Map classes to binary labels\n",
        "df['label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})\n",
        "df = df.dropna(subset=['Email Text', 'label']).reset_index(drop=True)\n",
        "\n",
        "print(\"Preprocessing email text (this may take a minute)...\")\n",
        "df['cleaned_text'] = df['Email Text'].apply(clean_email_text)\n",
        "df = df[df['cleaned_text'].str.strip() != ''].reset_index(drop=True)\n",
        "print(\"Preprocessing completed successfully!\")"
    ]
})

# Feature Engineering
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 4: Feature Extraction (TF-IDF Vectorization)\n",
        "We convert the processed text strings into numerical vectors using Term Frequency-Inverse Document Frequency (TF-IDF) with up to 5,000 features."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "X = df['cleaned_text']\n",
        "y = df['label']\n",
        "\n",
        "# 80% train, 20% test split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
        "print(f\"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}\")\n",
        "\n",
        "vectorizer = TfidfVectorizer(max_features=5000)\n",
        "X_train_tfidf = vectorizer.fit_transform(X_train)\n",
        "X_test_tfidf = vectorizer.transform(X_test)\n",
        "print(f\"TF-IDF Feature matrix shape: {X_train_tfidf.shape}\")"
    ]
})

# Model Training
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 5: Model Development & Evaluation\n",
        "We initialize and train four classifiers:\n",
        "1. **Logistic Regression**\n",
        "2. **Multinomial Naive Bayes**\n",
        "3. **Random Forest Classifier**\n",
        "4. **Multi-Layer Perceptron (Simple Neural Network)**"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "models = {\n",
        "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n",
        "    'Naive Bayes': MultinomialNB(),\n",
        "    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),\n",
        "    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, early_stopping=True)\n",
        "}\n",
        "\n",
        "results = {}\n",
        "y_preds = {}\n",
        "\n",
        "for name, model in models.items():\n",
        "    print(f\"Training {name}...\")\n",
        "    model.fit(X_train_tfidf, y_train)\n",
        "    \n",
        "    # Predict\n",
        "    y_pred = model.predict(X_test_tfidf)\n",
        "    y_preds[name] = y_pred\n",
        "    \n",
        "    # Metrics\n",
        "    acc = accuracy_score(y_test, y_pred)\n",
        "    prec = precision_score(y_test, y_pred, zero_division=0)\n",
        "    rec = recall_score(y_test, y_pred, zero_division=0)\n",
        "    f1 = f1_score(y_test, y_pred, zero_division=0)\n",
        "    \n",
        "    results[name] = {\n",
        "        'Accuracy': acc,\n",
        "        'Precision': prec,\n",
        "        'Recall': rec,\n",
        "        'F1-Score': f1\n",
        "    }\n",
        "    print(f\"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}\\n\")"
    ]
})

# Comparative Summary
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 6: Comparative Performance Table"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "results_df = pd.DataFrame(results).T\n",
        "display(results_df)"
    ]
})

# Performance visualization
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 7: Visualizations"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 1. ROC Curves\n",
        "plt.figure(figsize=(8, 6))\n",
        "for name, model in models.items():\n",
        "    if hasattr(model, \"predict_proba\"):\n",
        "        y_prob = model.predict_proba(X_test_tfidf)[:, 1]\n",
        "        fpr, tpr, _ = roc_curve(y_test, y_prob)\n",
        "        roc_auc = auc(fpr, tpr)\n",
        "        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')\n",
        "\n",
        "plt.plot([0, 1], [0, 1], 'k--')\n",
        "plt.xlim([0.0, 1.0])\n",
        "plt.ylim([0.0, 1.05])\n",
        "plt.xlabel('False Positive Rate')\n",
        "plt.ylabel('True Positive Rate')\n",
        "plt.title('ROC Curves Comparison')\n",
        "plt.legend(loc='lower right')\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.show()"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 2. Confusion Matrices\n",
        "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
        "axes = axes.ravel()\n",
        "\n",
        "for idx, (name, y_pred) in enumerate(y_preds.items()):\n",
        "    cm = confusion_matrix(y_test, y_pred)\n",
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,\n",
        "                xticklabels=['Safe', 'Phishing'],\n",
        "                yticklabels=['Safe', 'Phishing'])\n",
        "    axes[idx].set_title(f'Confusion Matrix: {name}')\n",
        "    axes[idx].set_ylabel('Actual')\n",
        "    axes[idx].set_xlabel('Predicted')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 3. Feature Importance (Logistic Regression Coefficients)\n",
        "lr_model = models['Logistic Regression']\n",
        "coefs = lr_model.coef_[0]\n",
        "features = vectorizer.get_feature_names_out()\n",
        "\n",
        "# Get top 20 positive coefficients (most indicative of Phishing)\n",
        "top_indices = coefs.argsort()[::-1][:20]\n",
        "top_features = [features[i] for i in top_indices]\n",
        "top_coefs = [coefs[i] for i in top_indices]\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "sns.barplot(x=top_coefs, y=top_features, hue=top_features, palette='rocket', legend=False)\n",
        "plt.title('Top 20 Keywords Indicating Phishing (Logistic Regression Coefficients)')\n",
        "plt.xlabel('Coefficient Value')\n",
        "plt.ylabel('Word')\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# Save notebook dict
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open('notebooks/phishing_detection.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Jupyter Notebook created at notebooks/phishing_detection.ipynb")
