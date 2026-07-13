import streamlit as st
import pandas as pd
import pickle
import os
import re

# Custom function imports (using absolute/relative import)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import preprocess_text

# Set page config
st.set_page_config(
    page_title="AI Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark theme styling with custom CSS
st.markdown("""
<style>
    /* Gradient headers */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(45deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: #A0AEC0;
        margin-bottom: 2rem;
    }
    /* Metric Card styling */
    .metric-card {
        background-color: #1A202C;
        border: 1px solid #2D3748;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #4A5568;
    }
    /* Alerts */
    .phish-alert {
        background-color: rgba(254, 178, 178, 0.15);
        border-left: 6px solid #FEB2B2;
        color: #FFF5F5;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .safe-alert {
        background-color: rgba(154, 230, 180, 0.15);
        border-left: 6px solid #9AE6B4;
        color: #F0FFF4;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_artifacts():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    # Check if models exist
    if not os.path.exists(models_dir):
        return None, None
        
    try:
        with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
            
        models = {}
        model_names = ['logistic_regression', 'naive_bayes', 'random_forest', 'neural_network']
        for name in model_names:
            path = os.path.join(models_dir, f"{name}_model.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    models[name.replace('_', ' ').title()] = pickle.load(f)
        return vectorizer, models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

vectorizer, models = load_artifacts()

# Sidebar Setup
st.sidebar.markdown("<h2 style='text-align: center;'>🛡️ Phishing Detector</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### About the Project")
st.sidebar.write(
    "This system uses Natural Language Processing (NLP) and Machine Learning "
    "to analyze email body text and classify it as **Safe** or **Phishing**."
)

st.sidebar.markdown("### Feature Indicators Analysis")
st.sidebar.info(
    "**1. Suspicious Keywords:** Urgency triggers (e.g. 'immediately', 'verify account').\n\n"
    "**2. Hyperlinks:** Phishing emails often contain high counts of URLs.\n\n"
    "**3. Text Cleanliness:** Preprocessing removes HTML noise and isolates linguistic features."
)

# Main Application Layout
st.markdown("<div class='main-title'>AI-Driven Phishing Email Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze raw emails instantly using advanced NLP and multiple machine learning models.</div>", unsafe_allow_html=True)

if vectorizer is None or not models:
    st.warning("⚠️ Trained models not found. Please wait for the training process to complete or run `download_and_train.py` to train the models and generate artifacts.")
else:
    # Model Selection
    selected_model_name = st.selectbox(
        "Select Machine Learning Model for Prediction:",
        list(models.keys())
    )
    model = models[selected_model_name]

    # Email Input
    email_input = st.text_area(
        "Paste the raw email content (including headers or body text):",
        height=250,
        placeholder="Dear customer, your bank account has been suspended. Please click on the link below to verify your login credentials immediately: http://login-secure-auth.com..."
    )

    if st.button("🔍 Analyze Email", type="primary"):
        if email_input.strip() == "":
            st.warning("Please paste some email text first.")
        else:
            # 1. Structural/Metadata Feature Extraction
            url_regex = re.compile(r'https?://\S+|www\.\S+')
            urls = url_regex.findall(email_input)
            url_count = len(urls)
            
            urgency_keywords = {'urgent', 'immediately', 'verify', 'suspend', 'action required', 'security alert', 'update your account', 'bank', 'password'}
            found_urgency = [kw for kw in urgency_keywords if kw in email_input.lower()]
            
            cleaned_email = preprocess_text(email_input)
            
            # 2. Vectorization
            input_vector = vectorizer.transform([cleaned_email])
            
            # 3. Prediction
            prediction = model.predict(input_vector)[0]
            
            # Calculate probability if possible
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_vector)[0][1]
            
            # Display results
            st.markdown("### Analysis Results")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if prediction == 1:
                    confidence = f"{prob * 100:.2f}%" if prob is not None else "High"
                    st.markdown(
                        f"<div class='phish-alert'>"
                        f"<h3>🚨 Warning: Phishing Email Detected!</h3>"
                        f"<p>Our classification model ({selected_model_name}) identified this email as a phishing attempt with <b>{confidence}</b> confidence.</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    confidence = f"{(1 - prob) * 100:.2f}%" if prob is not None else "High"
                    st.markdown(
                        f"<div class='safe-alert'>"
                        f"<h3>✅ Success: Legitimate Email (Safe)</h3>"
                        f"<p>Our classification model ({selected_model_name}) classified this email as safe/legitimate with <b>{confidence}</b> confidence.</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                # Preprocessing visualizer
                with st.expander("Show NLP Preprocessing Details (What the ML model saw)"):
                    st.markdown("**Original length:** " + str(len(email_input)) + " characters")
                    st.markdown("**Cleaned & Tokenized text:**")
                    st.code(cleaned_email if cleaned_email else "[No significant tokens found]")

            with col2:
                st.markdown("#### Key Indicators")
                
                # Indicators display
                if url_count > 0:
                    st.markdown(f"❌ **URLs Found:** {url_count}")
                    if url_count > 1:
                        st.caption("Multiple URLs are highly characteristic of phishing campaigns.")
                else:
                    st.markdown("✔️ **No URLs Found**")
                    
                if found_urgency:
                    st.markdown(f"⚠️ **Urgency/Security Keywords:** {', '.join(found_urgency)}")
                    st.caption("Urgency and fear-inducing keywords are commonly used in social engineering.")
                else:
                    st.markdown("✔️ **No urgency keywords detected**")
                    
                st.markdown(f"📝 **Word Count:** {len(email_input.split())}")
                
            # Extra model comparison
            st.markdown("---")
            st.markdown("#### Comparison Across Other Models")
            
            cols = st.columns(len(models))
            for idx, (m_name, m_obj) in enumerate(models.items()):
                m_pred = m_obj.predict(input_vector)[0]
                m_prob = m_obj.predict_proba(input_vector)[0][1] if hasattr(m_obj, "predict_proba") else None
                
                with cols[idx]:
                    status = "🚨 Phishing" if m_pred == 1 else "✅ Safe"
                    color = "#FEB2B2" if m_pred == 1 else "#9AE6B4"
                    pct = f"({m_prob*100:.1f}% Phish)" if m_prob is not None else ""
                    
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<h5>{m_name}</h5>"
                        f"<p style='color: {color}; font-weight: bold; font-size: 1.25rem;'>{status}</p>"
                        f"<span style='color: #A0AEC0; font-size: 0.85rem;'>{pct}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
