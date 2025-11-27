# ==========================
# 📧 Email Spam Classifier App
# Author: Muntaha Iftikhar
# ==========================

import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ========== LOAD MODEL & VECTORIZER ==========
model = pickle.load(open("svm_spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ========== TEXT PREPROCESSING FUNCTION ==========
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_email_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)
    cleaned = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(cleaned)

# ========== STREAMLIT PAGE CONFIG ==========
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== INNOVATIVE CSS STYLING ==========
st.markdown(
    """
    <style>

    /* ===========================
       🌌 NEBULA DARK AURA THEME
       =========================== */

    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 20% 20%, #1a002b, #080013, #000000);
        font-family: 'Inter', 'SF Pro Display', sans-serif;
    }

    /* 🎇 Floating Nebula Particles */
    @keyframes nebulaFloat {
        0%, 100% { transform: translateY(0px); opacity: 0.7; }
        50% { transform: translateY(-15px); opacity: 1; }
    }

    /* ✨ Neon Title */
    .main-title {
        font-size: 3.2rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #8a2be2, #e91ee3, #6a5acd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        text-shadow: 0 0 25px rgba(233, 30, 227, 0.4);
        animation: nebulaFloat 5s ease-in-out infinite;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #c7b7ff;
        margin-bottom: 2.5rem;
        letter-spacing: 0.05em;
    }

    /* 🟪 Glass Panel */
    .glass-container {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 25px;
        backdrop-filter: blur(25px);
        padding: 32px;
        margin-top: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 0 40px rgba(128, 0, 255, 0.15);
    }

    /* 📝 Text Area */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 18px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e8e0ff;
        font-size: 16px;
        transition: .3s ease;
    }

    .stTextArea textarea:focus {
        border: 2px solid #bb86fc;
        box-shadow: 0 0 15px rgba(187, 134, 252, 0.4);
    }

    /* 🔮 Neon Button */
    .stButton button {
        width: 100%;
        height: 60px;
        border-radius: 16px;
        background: linear-gradient(135deg, #6a0dad, #b026ff);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        border: none;
        transition: .4s ease;
        box-shadow: 0 8px 25px rgba(176, 38, 255, 0.4);
    }

    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(176, 38, 255, 0.6);
    }

    /* 🟣 Result Box */
    .result-box {
        padding: 2rem;
        border-radius: 22px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        color: #fff;
        animation: nebulaFloat 4s ease-in-out infinite;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 0 45px rgba(128, 0, 255, 0.25);
    }

    /* 📊 Stats Cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        text-align: center;
        transition: .3s ease;
    }

    .stat-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
    }

    .stat-value {
        font-size: 1.7rem;
        font-weight: 700;
        background: linear-gradient(90deg, #c77dff, #9d4edd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        color: #c5b2ff;
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }

    /* 🌙 Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.9rem;
        color: #9d8cff;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 1rem;
    }

    /* Hide Streamlit Branding */
    #MainMenu, footer, header { visibility: hidden; }

    </style>
    """,
    unsafe_allow_html=True
)


# ========== MAIN INTERFACE ==========
st.markdown('<h1 class="main-title">📧 Email Spam Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A Machine Learning App by Muntaha Iftikhar</p>', unsafe_allow_html=True)

# Stats Grid
st.markdown(
    """
    <div class="stats-grid">
        <div class="stat-card floating">
            <div class="stat-value">95.2%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-card floating" style="animation-delay: 0.2s">
            <div class="stat-value">SVM</div>
            <div class="stat-label">Algorithm</div>
        </div>
        <div class="stat-card floating" style="animation-delay: 0.4s">
            <div class="stat-value">TF-IDF</div>
            <div class="stat-label">Vectorization</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# # Glass container for main content
# st.markdown('<div class="glass-container">', unsafe_allow_html=True)

# Text input area
user_email = st.text_area("✉️ Paste your email content below:", height=200, placeholder="Type or paste email content here...")

if st.button("🔍 Analyze Email"):
    if user_email.strip() == "":
        st.warning("⚠️ Please enter some email text to analyze.")
    else:
        # Show custom loading animation
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem;">
                <div class="loading-dots">
                    <div></div><div></div><div></div><div></div>
                </div>
                <p style="color: #B0BEC5; margin-top: 1rem;">Analyzing email content...</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Preprocess and predict
        cleaned = clean_email_text(user_email)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.markdown('<div class="result-box" style="background: linear-gradient(135deg, #d9534f, #c9302c);">🚫 SPAM EMAIL DETECTED!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box" style="background: linear-gradient(135deg, #5cb85c, #449d44);">✅ This email is safe (Not Spam).</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown('<div class="footer">Developed by <b>Muntaha Iftikhar</b></div>', unsafe_allow_html=True)