import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import clean_text_pipeline
from feature_extraction import TfidfVectorizerScratch
from models_scratch import LogisticRegressionScratch, KNNClassifierScratch, NeuralNetworkScratch
from models_sklearn import get_sklearn_models

st.set_page_config(
    page_title="AI-Powered Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark/Modern Theme Styling
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    h1, h2, h3 {
        color: #6366f1 !important;
        font-family: 'Outfit', sans-serif;
    }
    .stButton>button {
        background-color: #6366f1;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4f46e5;
    }
</style>
""", unsafe_index=True)

@st.cache_resource
def load_and_train_pipeline():
    # Load dataset
    df = pd.read_csv("GlobalFakeNews_Research2026_v1.csv")
    X_raw = df['full_text'].values
    y = df['label'].values
    
    # Preprocess
    X_clean = np.array([clean_text_pipeline(text) for text in X_raw])
    
    # Vectorize
    vectorizer = TfidfVectorizerScratch(max_features=1000)
    X_vec = vectorizer.fit_transform(X_clean)
    
    # Train Models
    logreg = LogisticRegressionScratch(lr=0.5, n_iters=1000)
    logreg.fit(X_vec, y)
    
    knn = KNNClassifierScratch(n_neighbors=5, metric='cosine')
    knn.fit(X_vec, y)
    
    nn = NeuralNetworkScratch(input_size=X_vec.shape[1], hidden_size=32, lr=0.1, epochs=150)
    nn.fit(X_vec, y)
    
    return vectorizer, logreg, knn, nn, df

vectorizer, logreg, knn, nn, df = load_and_train_pipeline()

# Title Section
st.title("📰 AI-Powered Fake News Detection")
st.markdown("### IICT Summer Internship Program in AI & ML 2026 — Project-1")
st.markdown("---")

# Sidebar
st.sidebar.image("https://img.icons8.com/nolan/128/news.png", width=100)
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Live Prediction", "EDA Dashboard", "Model Performance & Benchmarks"])

if page == "Live Prediction":
    st.header("🔍 Real-time Article Verification")
    st.markdown("Type or paste a news article below to classify it as **Real** or **Fake** using our scratch ML models.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        title_input = st.text_input("Article Title", placeholder="Enter article headline here...")
        content_input = st.text_area("Article Content", placeholder="Paste article body here...", height=250)
        
        selected_model = st.selectbox("Select Classifier", ["Logistic Regression (Scratch)", "K-Nearest Neighbors (Scratch)", "Neural Network (Scratch)"])
        
        if st.button("Analyze News Article"):
            if not content_input.strip():
                st.error("Please enter some article content to analyze.")
            else:
                combined_text = f"{title_input} {content_input}"
                cleaned = clean_text_pipeline(combined_text)
                vectorized = vectorizer.transform([cleaned])
                
                # Predict
                if selected_model == "Logistic Regression (Scratch)":
                    pred = logreg.predict(vectorized)[0]
                    prob = logreg.predict_proba(vectorized)[0]
                    confidence = prob if pred == 1 else (1.0 - prob)
                elif selected_model == "K-Nearest Neighbors (Scratch)":
                    pred = knn.predict(vectorized)[0]
                    confidence = 1.0 # KNN returns discrete labels
                else: # Neural Net
                    prob = nn.predict_proba(vectorized)[0][0]
                    pred = 1 if prob >= 0.5 else 0
                    confidence = prob if pred == 1 else (1.0 - prob)
                
                # Display Results
                st.markdown("### Analysis Results")
                if pred == 0:
                    st.success(f"✅ **REAL NEWS** (Confidence: {confidence*100:.2f}%)")
                    st.info("Our models indicate this content aligns with credible reporting style and source benchmarks.")
                else:
                    st.error(f"🚨 **FAKE / MISINFORMATION** (Confidence: {confidence*100:.2f}%)")
                    st.warning("Warning: This content uses sensational, emotional, or non-traditional vocabulary patterns typical of clickbait.")
                    
    with col2:
        st.markdown("#### Sample Articles for Testing")
        st.markdown("**Real News Sample (Label 0):**")
        st.code("This article provides detailed analysis based on verified sources and official statements.", wrap_lines=True)
        
        st.markdown("**Fake News Sample (Label 1):**")
        st.code("This shocking revelation will completely change what you believe about recent events.", wrap_lines=True)

elif page == "EDA Dashboard":
    st.header("📊 Exploratory Data Analysis Dashboard")
    st.markdown("Visualizing metrics, scores, and text lengths of the **1,200 research articles** dataset.")
    
    tab1, tab2 = st.tabs(["Linguistic Properties", "Category Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("results/plots/class_distribution.png"):
                st.image("results/plots/class_distribution.png", caption="Label Distribution (Real vs Fake)")
            if os.path.exists("results/plots/scores_comparison.png"):
                st.image("results/plots/scores_comparison.png", caption="Sentiment and Clickbait Scores")
        with col2:
            if os.path.exists("results/plots/text_length_distribution.png"):
                st.image("results/plots/text_length_distribution.png", caption="Article Word Count Distribution")
            if os.path.exists("results/plots/trust_emotional_comparison.png"):
                st.image("results/plots/trust_emotional_comparison.png", caption="Trust Index vs Emotional Ratio")
                
    with tab2:
        if os.path.exists("results/plots/category_distribution.png"):
            st.image("results/plots/category_distribution.png", caption="Category breakdown of Real vs Fake articles", use_container_width=True)

elif page == "Model Performance & Benchmarks":
    st.header("🏆 Performance Comparison: Scratch vs. Scikit-Learn")
    st.markdown("Detailed breakdown of how our custom mathematical implementations from scratch compare to standard libraries.")
    
    if os.path.exists("results/plots/accuracy_comparison.png"):
        st.image("results/plots/accuracy_comparison.png", caption="Accuracy Comparison Chart")
        
    st.markdown("### Final Evaluated Metrics Table")
    if os.path.exists("results/model_results.json"):
        with open("results/model_results.json", "r") as f:
            metrics = json.load(f)
            
        data = []
        for model_name, info in metrics.items():
            data.append({
                "Model": model_name,
                "Accuracy": f"{info['Accuracy']:.4f}",
                "Precision": f"{info['Precision']:.4f}",
                "Recall": f"{info['Recall']:.4f}",
                "F1-Score": f"{info['F1-Score']:.4f}",
                "Training Time (s)": f"{info['TrainTime']:.4f}"
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True)
