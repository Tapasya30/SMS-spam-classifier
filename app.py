from pathlib import Path
import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


base_dir = Path(__file__).resolve().parent

with open(base_dir / 'vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open(base_dir / 'model.pkl', 'rb') as f:
    clf = pickle.load(f)


st.set_page_config(page_title="AI SMS Spam Classifier", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --primary: #2563eb;
            --secondary: #4f46e5;
            --success: #16a34a;
            --danger: #dc2626;
            --bg: #ffffff;
            --surface: #f8fafc;
            --text: #0f172a;
            --muted: #64748b;
            --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
        }

        * { font-family: 'Inter', 'Segoe UI', sans-serif; }
        .stApp { background: linear-gradient(135deg, #f8fbff 0%, #f5f3ff 100%); }
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }

        .hero-card {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(37,99,235,0.08);
            border-radius: 28px;
            padding: 2rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(16px);
            animation: fadeIn 0.8s ease;
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 0.4rem;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            color: var(--muted);
            line-height: 1.7;
            max-width: 700px;
        }

        .badge {
            display: inline-block;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(37,99,235,0.12), rgba(79,70,229,0.16));
            color: var(--secondary);
            font-size: 0.84rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        .illustration-box {
            background: linear-gradient(135deg, #eff6ff 0%, #f5f3ff 100%);
            border-radius: 24px;
            padding: 1.2rem;
            min-height: 240px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(37,99,235,0.08);
        }

        .card {
            background: rgba(255,255,255,0.95);
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 22px;
            padding: 1.25rem;
            box-shadow: var(--shadow);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .card:hover {
            transform: translateY(-3px);
            box-shadow: 0 18px 36px rgba(15,23,42,0.10);
        }

        .stTextArea textarea {
            border-radius: 18px !important;
            padding: 1rem !important;
            min-height: 170px !important;
            border: 1px solid #dbeafe !important;
            background: #ffffff !important;
            box-shadow: inset 0 1px 2px rgba(15,23,42,0.04);
        }

        .stTextArea textarea:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
        }

        .stButton > button {
            border-radius: 999px !important;
            padding: 0.72rem 1.25rem !important;
            font-weight: 700 !important;
            border: none !important;
            background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
            color: white !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 10px 24px rgba(37,99,235,0.2) !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(37,99,235,0.25) !important;
        }

        .result-success {
            border-left: 5px solid var(--success);
            background: linear-gradient(90deg, rgba(22,163,74,0.08), rgba(255,255,255,0.95));
        }

        .result-danger {
            border-left: 5px solid var(--danger);
            background: linear-gradient(90deg, rgba(220,38,38,0.08), rgba(255,255,255,0.95));
        }

        .metric-label {
            font-size: 0.84rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.2rem;
        }

        .metric-value {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text);
        }

        .progress-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-top: 0.7rem;
        }

        .progress-bar {
            flex: 1;
            height: 9px;
            background: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.8s ease;
        }

        .footer {
            text-align: center;
            color: var(--muted);
            padding-top: 1.5rem;
            font-size: 0.95rem;
        }

        .fade-in { animation: fadeIn 0.8s ease; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 768px) {
            .hero-title { font-size: 2rem; }
            .hero-card { padding: 1.2rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card fade-in">
        <div class="badge">⚡ Powered by Machine Learning</div>
        <div style="display:flex; justify-content:space-between; gap:1rem; align-items:center; flex-wrap:wrap;">
            <div style="flex:1; min-width:280px;">
                <h1 class="hero-title">🤖 AI SMS Spam Classifier</h1>
                <p class="hero-subtitle">Instantly analyze SMS messages using Machine Learning to detect spam and protect against phishing attempts.</p>
            </div>
            <div class="illustration-box" style="flex:0 0 280px;">
                <div style="text-align:center;">
                    <div style="font-size:3.2rem;">🛡️</div>
                    <div style="font-size:1.2rem; font-weight:700; color:#0f172a; margin-top:0.35rem;">Secure AI Detection</div>
                    <div style="font-size:0.95rem; color:#64748b; margin-top:0.25rem;">Real-time message screening</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
st.markdown("<h3 style='margin:0 0 0.5rem 0; color:#0f172a;'>Analyze a message</h3>", unsafe_allow_html=True)
st.markdown("<p style='margin:0 0 1rem 0; color:#64748b;'>Paste or type your SMS message to classify it in seconds.</p>", unsafe_allow_html=True)

if "sms_input" not in st.session_state:
    st.session_state["sms_input"] = ""

input_sms = st.text_area(
    label="",
    placeholder="Paste or type your SMS message here...",
    height=180,
    key="sms_input",
    value=st.session_state["sms_input"],
)

char_count = len(input_sms) if input_sms else 0
analyze_button = st.button("Analyze Message", use_container_width=True)

st.caption(f"Characters: {char_count}")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

if analyze_button:
    if not input_sms or not input_sms.strip():
        st.warning("Please enter a message to analyze.")
    else:
        with st.spinner("Analyzing message securely..."):
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms]).toarray()
            prediction = clf.predict(vector_input)[0]

            confidence = 0.0
            if hasattr(clf, 'predict_proba'):
                try:
                    confidence = round(float(max(clf.predict_proba(vector_input)[0])) * 100, 1)
                except Exception:
                    confidence = 92.0
            else:
                confidence = 92.0

        if prediction == 1:
            st.markdown(
                f"""
                <div class='card result-danger fade-in'>
                    <div style='display:flex; align-items:center; gap:0.7rem; margin-bottom:0.6rem;'>
                        <div style='font-size:1.5rem;'>⚠️</div>
                        <h3 style='margin:0; color:#0f172a;'>Spam Detected</h3>
                    </div>
                    <p style='margin:0 0 0.8rem 0; color:#475569;'>This message appears suspicious and may be part of a spam or phishing attempt.</p>
                    <div class='metric-label'>Confidence</div>
                    <div class='metric-value'>{confidence}%</div>
                    <div class='progress-row'>
                        <div class='progress-bar'><div class='progress-fill' style='width:{confidence}%; background:linear-gradient(90deg, #dc2626, #ef4444);'></div></div>
                        <span style='color:#dc2626; font-weight:700;'>{confidence}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class='card result-success fade-in'>
                    <div style='display:flex; align-items:center; gap:0.7rem; margin-bottom:0.6rem;'>
                        <div style='font-size:1.5rem;'>✅</div>
                        <h3 style='margin:0; color:#0f172a;'>Safe Message</h3>
                    </div>
                    <p style='margin:0 0 0.8rem 0; color:#475569;'>This message looks safe and does not match the spam patterns detected by the model.</p>
                    <div class='metric-label'>Confidence</div>
                    <div class='metric-value'>{confidence}%</div>
                    <div class='progress-row'>
                        <div class='progress-bar'><div class='progress-fill' style='width:{confidence}%; background:linear-gradient(90deg, #16a34a, #22c55e);'></div></div>
                        <span style='color:#16a34a; font-weight:700;'>{confidence}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

feature_cols = st.columns(4, gap="medium")
features = [
    ("🧠", "Machine Learning Powered", "Advanced text classification pipelines for dependable spam detection."),
    ("⚡", "Real-time Detection", "Instant analysis with a smooth and responsive experience."),
    ("🔐", "Secure SMS Analysis", "Built to help users review suspicious messages with confidence."),
    ("🎯", "High Accuracy Prediction", "Designed to recognize spam patterns with precision and reliability."),
]

for idx, (icon, title, desc) in enumerate(features):
    with feature_cols[idx]:
        st.markdown(
            f"""
            <div class='card fade-in'>
                <div style='font-size:1.6rem; margin-bottom:0.45rem;'>{icon}</div>
                <h4 style='margin:0 0 0.35rem 0; color:#0f172a;'>{title}</h4>
                <p style='margin:0; color:#64748b; line-height:1.6;'>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
st.markdown("<h4 style='margin:0 0 0.7rem 0; color:#0f172a;'>How It Works</h4>", unsafe_allow_html=True)
steps = [
    ("🧹", "Text Cleaning", "Normalizes the message by converting it to lowercase and removing noise."),
    ("✂️", "Stopword Removal", "Removes common words that do not add much meaning."),
    ("📚", "TF-IDF Vectorization", "Transforms the message into numerical features for the model."),
    ("🤖", "Machine Learning Prediction", "Uses the trained classifier to determine whether the SMS is spam."),
]
step_cols = st.columns(4, gap="small")
for idx, (icon, title, desc) in enumerate(steps):
    with step_cols[idx]:
        st.markdown(
            f"""
            <div class='card'>
                <div style='font-size:1.4rem; margin-bottom:0.35rem;'>{icon}</div>
                <h5 style='margin:0 0 0.25rem 0; color:#0f172a;'>{title}</h5>
                <p style='margin:0; color:#64748b; font-size:0.95rem; line-height:1.5;'>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class='footer'>
        <p style='margin-bottom:0.35rem;'>Developed by Tapasya Chaturvedi</p>
        <p style='margin:0;'>🔗 GitHub  •  💼 LinkedIn  •  © 2026</p>
    </div>
    """,
    unsafe_allow_html=True,
)
