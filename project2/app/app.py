import streamlit as st
import pandas as pd
import pickle
import os
import re
import sys

# Import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import preprocess_text as clean_phish_text
from src.fake_news_preprocessing import clean_news_text
from src.news_scrapper import fetch_live_news

# Set page config
st.set_page_config(
    page_title="AI Security & Information Integrity Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium dark theme CSS
st.markdown("""
<style>
    /* Gradient titles */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(45deg, #4F46E5, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .main-title-fake {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(45deg, #F59E0B, #EF4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 1.5rem;
    }
    /* Metric Card */
    .metric-card {
        background-color: #1A202C;
        border: 1px solid #2D3748;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #4E5D78;
    }
    /* Alerts */
    .phish-alert {
        background-color: rgba(239, 68, 68, 0.15);
        border-left: 6px solid #EF4444;
        color: #FEE2E2;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .safe-alert {
        background-color: rgba(16, 185, 129, 0.15);
        border-left: 6px solid #10B981;
        color: #D1FAE5;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    /* News Cards */
    .news-card {
        background-color: #1F2937;
        border: 1px solid #374151;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
    }
    .fake-badge {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid #EF4444;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .real-badge {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid #10B981;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Path setup
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Cache artifacts
@st.cache_resource
def load_phishing_artifacts():
    try:
        with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        models = {}
        names = ['logistic_regression', 'naive_bayes', 'random_forest', 'neural_network']
        for name in names:
            path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    models[name.replace('_', ' ').title()] = pickle.load(f)
        return vectorizer, models
    except Exception as e:
        return None, None

@st.cache_resource
def load_fake_news_artifacts():
    try:
        with open(os.path.join(MODELS_DIR, 'fake_news_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        models = {}
        names = ['knn', 'logistic_regression', 'random_forest', 'neural_network']
        for name in names:
            path = os.path.join(MODELS_DIR, f"fake_news_{name}_model.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    models[name.replace('_', ' ').title()] = pickle.load(f)
        return vectorizer, models
    except Exception as e:
        return None, None

# Load all
phish_vec, phish_models = load_phishing_artifacts()
news_vec, news_models = load_fake_news_artifacts()

# Sidebar Navigation
st.sidebar.markdown("<h2 style='text-align: center;'>🛡️ IICT Suite</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Select Analysis Module:",
    ["🛡️ Phishing Email Detector", "📰 Live Fake News Analyzer"]
)

# ----------------- PHISHING DETECTOR PAGE -----------------
if app_mode == "🛡️ Phishing Email Detector":
    st.markdown("<div class='main-title'>AI-Driven Phishing Email Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Inspect email text and structural indicators to flag malicious social engineering attempts.</div>", unsafe_allow_html=True)
    
    if phish_vec is None or not phish_models:
        st.warning("⚠️ Phishing model checkpoints not found. Run `download_and_train.py` first.")
    else:
        selected_model = st.selectbox("Select ML Classifier:", list(phish_models.keys()), index=3)
        model = phish_models[selected_model]
        
        email_input = st.text_area(
            "Paste email content here:",
            height=200,
            placeholder="Dear user, your credentials have expired. Verify immediately at http://login-secure-auth.com..."
        )
        
        if st.button("🔍 Analyze Email", type="primary"):
            if email_input.strip() == "":
                st.warning("Please paste some content first.")
            else:
                # Features
                url_regex = re.compile(r'https?://\S+|www\.\S+')
                urls = url_regex.findall(email_input)
                url_count = len(urls)
                
                urgency_keywords = {'urgent', 'immediately', 'verify', 'suspend', 'action required', 'security alert', 'update'}
                found_urgency = [w for w in urgency_keywords if w in email_input.lower()]
                
                cleaned = clean_phish_text(email_input)
                vec_input = phish_vec.transform([cleaned])
                
                # Predict
                pred = model.predict(vec_input)[0]
                prob = model.predict_proba(vec_input)[0][1] if hasattr(model, "predict_proba") else None
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    if pred == 1:
                        conf = f"{prob*100:.2f}%" if prob is not None else "High"
                        st.markdown(
                            f"<div class='phish-alert'>"
                            f"<h3>🚨 Warning: Phishing Attempt Identified!</h3>"
                            f"<p>Flagged by <b>{selected_model}</b> with <b>{conf}</b> confidence.</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        conf = f"{(1-prob)*100:.2f}%" if prob is not None else "High"
                        st.markdown(
                            f"<div class='safe-alert'>"
                            f"<h3>✅ Legitimate Email (Safe)</h3>"
                            f"<p>Classified as safe by <b>{selected_model}</b> with <b>{conf}</b> confidence.</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    with st.expander("Show Cleaned Tokens"):
                        st.code(cleaned if cleaned else "[No tokens]")
                        
                with col2:
                    st.markdown("#### Key Indicators")
                    if url_count > 0:
                        st.write(f"❌ **URLs Found:** {url_count}")
                    else:
                        st.write("✔️ **No URLs found**")
                    if found_urgency:
                        st.write(f"⚠️ **Urgency keywords:** {', '.join(found_urgency)}")
                    else:
                        st.write("✔️ **No urgency keywords**")
                    st.write(f"📝 **Word Count:** {len(email_input.split())}")
                    
                # Other Models comparison
                st.markdown("---")
                st.markdown("#### Model Predictions Comparison")
                cols = st.columns(len(phish_models))
                for idx, (m_name, m_obj) in enumerate(phish_models.items()):
                    m_pred = m_obj.predict(vec_input)[0]
                    m_prob = m_obj.predict_proba(vec_input)[0][1] if hasattr(m_obj, "predict_proba") else None
                    with cols[idx]:
                        status = "🚨 Phishing" if m_pred == 1 else "✅ Safe"
                        color = "#EF4444" if m_pred == 1 else "#10B981"
                        pct = f"({m_prob*100:.1f}%)" if m_prob is not None else ""
                        st.markdown(
                            f"<div class='metric-card'>"
                            f"<h5>{m_name}</h5>"
                            f"<p style='color: {color}; font-weight: bold; font-size: 1.2rem;'>{status}</p>"
                            f"<span style='color: #A0AEC0; font-size: 0.8rem;'>{pct}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

# ----------------- FAKE NEWS ANALYZER PAGE -----------------
else:
    st.markdown("<div class='main-title-fake'>AI-Powered Fake News Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Analyze news headlines or scrape live international news feeds using our classification suite.</div>", unsafe_allow_html=True)
    
    if news_vec is None or not news_models:
        st.warning("⚠️ Fake news model checkpoints not found. Run `train_fake_news.py` first.")
    else:
        selected_model = st.selectbox("Select ML Classifier:", list(news_models.keys()), index=3)
        model = news_models[selected_model]
        
        tab1, tab2 = st.tabs(["✍️ Analyze Custom News", "🌍 Scrape Live Headlines"])
        
        with tab1:
            headline = st.text_input("Enter News Headline:")
            body = st.text_area("Enter Article Content:", height=180)
            
            if st.button("🔍 Analyze News Content", type="primary"):
                if headline.strip() == "" and body.strip() == "":
                    st.warning("Please enter some text.")
                else:
                    combined = f"{headline}. {body}"
                    cleaned = clean_news_text(combined)
                    vec_input = news_vec.transform([cleaned])
                    
                    pred = model.predict(vec_input)[0]
                    prob = model.predict_proba(vec_input)[0][1] if hasattr(model, "predict_proba") else None
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        if pred == 1:
                            conf = f"{prob*100:.2f}%" if prob is not None else "High"
                            st.markdown(
                                f"<div class='phish-alert'>"
                                f"<h3>🚨 Warning: Likely Fake News!</h3>"
                                f"<p>Classified as <b>FAKE</b> by <b>{selected_model}</b> with <b>{conf}</b> confidence.</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            conf = f"{(1-prob)*100:.2f}%" if prob is not None else "High"
                            st.markdown(
                                f"<div class='safe-alert'>"
                                f"<h3>✅ Likely Real / Legitimate News</h3>"
                                f"<p>Classified as <b>REAL</b> by <b>{selected_model}</b> with <b>{conf}</b> confidence.</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        with st.expander("Show Preprocessed Tokens"):
                            st.code(cleaned if cleaned else "[No tokens]")
                    with col2:
                        st.markdown("#### Meta Analysis")
                        st.write(f"**Original Word Count:** {len(combined.split())}")
                        st.write(f"**Cleaned Word Count:** {len(cleaned.split())}")
                        
                    # Compare other models
                    st.markdown("---")
                    st.markdown("#### Model Predictions Comparison")
                    cols = st.columns(len(news_models))
                    for idx, (m_name, m_obj) in enumerate(news_models.items()):
                        m_pred = m_obj.predict(vec_input)[0]
                        m_prob = m_obj.predict_proba(vec_input)[0][1] if hasattr(m_obj, "predict_proba") else None
                        with cols[idx]:
                            status = "🚨 Fake" if m_pred == 1 else "✅ Real"
                            color = "#EF4444" if m_pred == 1 else "#10B981"
                            pct = f"({m_prob*100:.1f}%)" if m_prob is not None else ""
                            st.markdown(
                                f"<div class='metric-card'>"
                                f"<h5>{m_name}</h5>"
                                f"<p style='color: {color}; font-weight: bold; font-size: 1.2rem;'>{status}</p>"
                                f"<span style='color: #A0AEC0; font-size: 0.8rem;'>{pct}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            
        with tab2:
            st.markdown("### Live NewsAPI Scraper Integration")
            category = st.selectbox(
                "Select Headline Category:",
                ["General", "Technology", "Business", "Science", "Health", "Sports", "Entertainment"]
            )
            keyword = st.text_input("Or query by specific keyword:")
            
            if st.button("Fetch & Classify News", type="secondary"):
                with st.spinner("Fetching news..."):
                    q = keyword.strip() if keyword.strip() != "" else None
                    articles = fetch_live_news(query=q, category=category.lower())
                    
                    if not articles:
                        st.warning("No articles returned from NewsAPI.")
                    else:
                        st.success(f"Fetched {len(articles)} articles. Classifying using {selected_model}...")
                        
                        for art in articles:
                            cleaned_art = clean_news_text(art["full_text"])
                            if cleaned_art.strip() == "":
                                continue
                                
                            vec_art = news_vec.transform([cleaned_art])
                            pred_art = model.predict(vec_art)[0]
                            prob_art = model.predict_proba(vec_art)[0][1] if hasattr(model, "predict_proba") else None
                            
                            badge = ""
                            if pred_art == 1:
                                pct = f"({prob_art*100:.1f}% Phony)" if prob_art is not None else ""
                                badge = f"<span class='fake-badge'>🚨 FAKE {pct}</span>"
                            else:
                                pct = f"({(1-prob_art)*100:.1f}% Real)" if prob_art is not None else ""
                                badge = f"<span class='real-badge'>✅ REAL {pct}</span>"
                                
                            st.markdown(
                                f"<div class='news-card'>"
                                f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>"
                                f"<h4 style='margin: 0;'>{art['title']}</h4>"
                                f"{badge}"
                                f"</div>"
                                f"<p style='color: #9CA3AF; font-size: 0.9rem; margin-bottom: 0.5rem;'>{art['description'] or 'No description available.'}</p>"
                                f"<div style='font-size: 0.8rem; color: #6B7280;'>"
                                f"Source: <b>{art['source']}</b> | <a href='{art['url']}' target='_blank' style='color: #3B82F6;'>Full Article ↗</a>"
                                f"</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
