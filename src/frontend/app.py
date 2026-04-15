"""
app.py  —  Streamlit Frontend
==============================
Pages:
  🏠 Home
  🔍 Analyze Text     ← paste / type text
  📷 Camera & Image   ← webcam capture + image upload → OCR → analyze
  📊 Dashboard
  🏆 Model Results

Run: streamlit run frontend/app.py
"""

import sys, os, io, base64, requests, subprocess, socket, time, re
from urllib.parse import urlparse
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Root path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'core'))

API = os.environ.get("API_URL", "http://localhost:8000")
PROC_DIR = os.path.join(ROOT, 'data', 'processed')
SAVED_MODELS = os.path.join(ROOT, 'saved_models')

# MONOLITHIC MODE: Load predictor directly for speed and Cloud compatibility
def get_predictor():
    import importlib
    import models.predict
    importlib.reload(models.predict)
    from models.predict import PoeticsPredictor
    return PoeticsPredictor()

predictor = get_predictor()

st.set_page_config(
    page_title="Computational Poetics Analyzer",
    page_icon="📜", layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Playfair+Display:ital,wght@0,700;1,700&display=swap');

:root {
    --bg-dark: #0f1116;
    --glass-bg: rgba(255, 255, 255, 0.03);
    --glass-border: rgba(255, 255, 255, 0.08);
    --accent-primary: #6366f1;
    --accent-secondary: #a5b4fc;
    --text-main: #e2e8f0;
    --text-dim: #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text-main);
}

.stApp {
    background: radial-gradient(circle at 0% 0%, #1e1b4b 0%, #0f172a 50%, #0f1116 100%);
}

/* Glass Container Base - Targeting Streamlit's native container with border=True */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border) !important;
    border-radius: 24px !important;
    padding: 1rem;
    margin: 1.5rem 0;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

/* Header / Hero Section */
.hero {
    text-align: center;
    padding: 4rem 2rem;
    background: linear-gradient(180deg, rgba(99, 102, 241, 0.05) 0%, transparent 100%);
    border-radius: 32px;
    margin-bottom: 3rem;
    border: 1px solid rgba(99, 102, 241, 0.1);
}

.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 800;
    margin-bottom: 1rem;
    background: linear-gradient(to right, #ffffff, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    font-size: 1.25rem;
    color: var(--text-dim);
    letter-spacing: 0.5px;
}

/* Metric Cards */
.card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(8px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
}

.card:hover {
    transform: translateY(-8px);
    background: rgba(99, 102, 241, 0.06);
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
}

.card h3 {
    color: var(--accent-secondary);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1rem;
}

.card .val {
    font-size: 2.2rem;
    font-weight: 800;
    color: #fff;
    margin-bottom: 0.5rem;
}

.card .sub {
    font-size: 0.8rem;
    color: var(--text-dim);
}

/* Poem Box */
.poembox {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--glass-border);
    border-left: 4px solid var(--accent-primary);
    padding: 2rem;
    border-radius: 12px;
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    line-height: 1.8;
    color: #f8fafc;
    margin: 2rem 0;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.2);
}

/* Stylometrics Grid */
.stylo-card {
    background: rgba(15, 23, 42, 0.3);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}

.stylo-card:hover {
    background: rgba(99, 102, 241, 0.1);
    border-color: var(--accent-primary);
}

.stylo-label {
    font-size: 0.7rem;
    color: var(--accent-secondary);
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.stylo-val {
    font-size: 1.75rem;
    font-weight: 800;
    color: #fff;
}

/* Tooltips & Tips */
.tip {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    padding: 1rem 1.5rem;
    border-radius: 12px;
    color: #c7d2fe;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

/* OCR Box */
.ocr-box {
    background: #000;
    color: #22c55e;
    font-family: 'Courier New', monospace;
    padding: 1.5rem;
    border-radius: 8px;
    border: 1px solid #14532d;
    white-space: pre-wrap;
    margin-bottom: 1rem;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background-color: #0c0e12 !important;
    border-right: 1px solid var(--glass-border);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.4rem 0.6rem !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.2px !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.3s ease !important;
    white-space: nowrap !important;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero, .card, .glass-panel, .poembox {
    animation: fadeIn 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) backwards;
}

.card:nth-child(1) { animation-delay: 0.1s; }
.card:nth-child(2) { animation-delay: 0.2s; }
.card:nth-child(3) { animation-delay: 0.3s; }
.card:nth-child(4) { animation-delay: 0.4s; }

/* Custom Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
EMOTION_EMOJI = {
    'anger':'😠','sadness':'😢','fear':'😨','joy':'😊',
    'love':'❤️','peace':'😌','courage':'💪','surprise':'😲'
}
GENRE_COLOR = {
    'tragedy':'#2C3E50','gothic':'#6B2D8B','horror':'#E74C3C','modern':'#3498DB'
}

SAMPLE_POEMS = {
    "🧟 Gothic": """Once upon a midnight dreary, while I pondered, weak and weary,
Over many a quaint and curious volume of forgotten lore—
While I nodded, nearly napping, suddenly there came a tapping,
As of some one gently rapping, rapping at my chamber door.""",
    "🏙️ Modern": """Let us go then, you and I,
When the evening is spread out against the sky
Like a patient etherized upon a table;
Let us go, through certain half-deserted streets,
The muttering retreats
Of restless nights in one-night cheap hotels
And sawdust restaurants with oyster-shells""",
    "🌿 Nature": """I wandered lonely as a cloud
That floats on high o'er vales and hills,
When all at once I saw a crowd,
A host, of golden daffodils;
Beside the lake, beneath the trees,
Fluttering and dancing in the breeze.""",
    "⚔️ War": """Gas! GAS! Quick, boys!—An ecstasy of fumbling,
Fitting the clumsy helmets just in time,
But some one still was yelling out and stumbling
And flound’ring like a man in fire or lime.—
Dim through the misty panes and thick green light,
As under a green sea, I saw him drowning.""",
    "❤️ Romance": """Bright star, would I were stedfast as thou art—
Not in lone splendour hung aloft the night
And watching, with eternal lids apart,
Like nature's patient, sleepless Eremite,
The moving waters at their priestlike task""",
    "🎭 Tragedy": """Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day,
To the last syllable of recorded time;
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!""",
    "💀 Horror": """The shadows in the corner began to breathe, 
A cold, rhythmic pulse that chilled the marrow. 
No moon light pierced the heavy velvet drapes, 
Only the sound of something wet dragging itself
Across the floorboards, closer with every beat 
Of my terrified, failing heart."""
}


# ─── API helpers ──────────────────────────────────────────────────────────────
def call_analyze(text: str) -> dict:
    """Monolithic call directly to the inference engine."""
    try:
        return predictor.analyze(text)
    except Exception as e:
        return {"error": str(e)}


def call_upload_image(file_bytes: bytes, filename: str) -> dict:
    """Monolithic OCR call — handles local Tesseract extraction."""
    try:
        from ocr.image_to_text import extract_from_bytes
        text = extract_from_bytes(file_bytes)
        if not text.strip():
            return {"error": "No text extracted from image."}
        res = predictor.analyze(text)
        res['ocr_extracted_text'] = text
        return res
    except Exception as e:
        return {"error": f"OCR extraction failed: {str(e)}"}


# ─── Data Helpers ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_genre_csv():
    p = os.path.join(PROC_DIR, 'genre_clean.csv')
    return pd.read_csv(p) if os.path.exists(p) else None


# ─── Result Display ───────────────────────────────────────────────────────────
def show_results(result: dict, expert: bool):
    if 'error' in result:
        st.error(f"❌ {result['error']}")
        if 'connect' in result['error'].lower():
            st.info("Make sure the backend is running:\n```\nuvicorn backend.api:app --reload --port 8000\n```")
        return

    # ── 5 metric cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    genre = (result.get('genre') or 'N/A').title()
    fig   = (result.get('figurative_type') or 'N/A').title()
    emo   = result.get('emotion') or 'N/A'
    score = result.get('quality_score')
    grade = result.get('grade', '')
    emoji = EMOTION_EMOJI.get(emo, '🎭')

    with c1:
        st.markdown(f"""<div class="card">
            <h3>🎭 Genre</h3>
            <p class="val">{genre}</p>
            <p class="sub">Literary category</p></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card">
            <h3>🎨 Figurative Language</h3>
            <p class="val">{fig}</p>
            <p class="sub">Primary device</p></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="card">
            <h3>💭 Emotion</h3>
            <p class="val">{emoji} {emo.title()}</p>
            <p class="sub">Dominant feeling</p></div>""",
            unsafe_allow_html=True)
    with c4:
        score_str = f"{score}/10" if score is not None else 'N/A'
        st.markdown(f"""<div class="card">
            <h3>⭐ Quality Score</h3>
            <p class="val">{score_str}</p>
            <p class="sub">{grade}</p></div>""",
            unsafe_allow_html=True)

    # ── Stylometric Fingerprint (Now default) ──────────────────────────────────
    stylo = result.get('stylometrics', {})
    if stylo:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📐 Stylometric Fingerprint")
        
        label_map = [
            ('word_count',             '📝 Word Count',         'Total tokens'),
            ('vocabulary_richness',    '🔤 Vocab Richness',     'Unique/Total ratio'),
            ('avg_word_length',        '📏 Avg Word Length',    'Chars per word'),
            ('line_count',             '📄 Line Count',         'Total verses'),
            ('avg_line_length',        '📐 Avg Line Length',    'Words per line'),
            ('punctuation_density',    '✍️ Punctuation',        'Symbol frequency'),
            ('avg_syllables_per_word', '🔊 Avg Syllables',      'Rhythmic complexity'),
            ('strong_vocab_ratio',     '📈 Lexical Density',    'Strong Vocab Ratio'),
        ]
        
        # Grid layout for stylometrics
        cols = st.columns(4)
        for i, (k, label, sub) in enumerate(label_map):
            val = stylo.get(k, 'N/A')
            if isinstance(val, float):
                val = f"{val:.3f}" if 'richness' in k or 'density' in k else f"{val:.2f}"
            
            with cols[i % 4]:
                st.markdown(f"""
                    <div class="stylo-card">
                        <div class="stylo-label">{label}</div>
                        <div class="stylo-val">{val}</div>
                        <div class="stylo-sub">{sub}</div>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
        <div class="hero">
            <h1>The Art of Computational Poetics</h1>
            <p>Unveiling the hidden structures of verse through Neural Logic & Linguistics</p>
        </div>
    """, unsafe_allow_html=True)

    # Monolithic Status Check
    n_loaded = sum(1 for m in [predictor.genre_model, predictor.fig_model, 
                                predictor.emo_model, predictor.qual_model] if m)
    
    if n_loaded == 4:
        st.markdown('<div class="tip">✅ <strong>All Neural Engines Online:</strong> System operating at peak performance.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="tip">⚠️ <strong>Initialization in Progress:</strong> {n_loaded}/4 engines active. Some metrics may be unavailable.</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    features = [
        ("🎭", "Genre", "Categorizes poems into War, Nature, Romance, Tragedy, Gothic, Horror, or Modern."),
        ("🎨", "Figurative", "Identifies Similes, Metaphors, Sarcasm, Idioms, Personifications, and Hyperboles."),
        ("💭", "Emotion", "Detects core sentiments: Joy, Sadness, Fear, Love, Peace, Courage, or Surprise."),
        ("⭐", "Quality", "A 0–10 quality score based on structural complexity and stylistic richness.")
    ]
    
    for col, (icon, title, desc) in zip([c1,c2,c3,c4], features):
        with col:
            st.markdown(f"""
                <div class="card">
                    <div style="font-size:3rem;margin-bottom:1rem">{icon}</div>
                    <div class="val" style="font-size:1.2rem; margin-bottom: 0.5rem;">{title}</div>
                    <div class="sub">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    with st.expander("📖 Deep Technical Architecture (Neural Era)"):
        st.markdown("""
### Multimodal Analysis Pipeline
The Poetics Analyzer employs a tiered architecture to process verse:

1.  **Linguistic Layer**: Extracts 16 stylometric features including structural variance, lexical density, and theme-specific markers.
2.  **Semantic Layer**: Utilizes `all-MiniLM-L6-v2` transformer embeddings to capture high-dimensional poetic sentiment.
3.  **Inference Layer**: An ensemble of **XGBoost**, **Bi-LSTM**, and **Deep MLP** models collaborating to provide consistent predictions.
4.  **Vision Layer**: Robust OCR powered by **EasyOCR** and **Tesseract** for physical text processing.

### Datasets & Training
| Dimension | Source | Volume | Optimization |
|:---:|:---|:---|:---|
| **Genre** | Balanced Poetry Dataset | 7,693 samples | Stratified Cross-Validation |
| **Figurative** | Hybrid NLP Dataset | 9,484 samples | Random Oversampling |
| **Emotion** | Annotated Poem corpus | 1,811 samples | Synthetic Minority Weighted |
| **Quality** | Quality Scored Corpus | 868 samples | Ridge-MLP Regression |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — ANALYZE TEXT
# ══════════════════════════════════════════════════════════════════════════════
def page_analyze(expert: bool):
    st.markdown("""
        <div class="hero">
            <h1>🔍 Analytical Deep Dive</h1>
            <p>Paste your verses for comprehensive structural & thematic audit</p>
        </div>
    """, unsafe_allow_html=True)

    default = st.session_state.get('sample_text', '')
    prefill = st.session_state.pop('prefill', '')
    if prefill: default = prefill

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        text = st.text_area("Source Text:", value=default, height=230,
                                placeholder="Paste a poem or any creative text here...")
        
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            btn = st.button("🔍 Run Analysis", type="primary", use_container_width=True)

    if btn:
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Decoding Stylometrics..."):
                result = call_analyze(text)
            
            st.markdown(f'<div class="poembox">{text}</div>', unsafe_allow_html=True)
            st.markdown("### 📊 Metrics & Insight")
            show_results(result, expert)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — CAMERA & IMAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_camera(expert: bool):
    st.markdown("""
        <div class="hero">
            <h1>📷 Vision & OCR Module</h1>
            <p>Process handwritten or printed verse via Webcam or Image Upload</p>
        </div>
    """, unsafe_allow_html=True)

    # OCR Engine Status Tracking
    try:
        from ocr.image_to_text import OCR_CONFIG
        status_parts = []
        if OCR_CONFIG['EASY_OK']: status_parts.append('<span style="color:#22c55e">● EasyOCR</span>')
        if OCR_CONFIG['TESSERACT_OK']: status_parts.append('<span style="color:#22c55e">● Tesseract</span>')
        status_html = " | ".join(status_parts) if status_parts else '<span style="color:#ef4444">● All Engines Offline</span>'
    except:
        status_html = '<span style="color:#94a3b8">● Status Unknown</span>'
    
    st.markdown(f'<div class="tip"><strong>Engine Protocol:</strong> {status_html}</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📷 Webcam Interface", "📁 Static Image Upload"])

    # ── TAB 1: WEBCAM ─────────────────────────────────────────────────────────
    with tab1:
        with st.container(border=True):
            st.info("💡 **Instructions:** Frame the text clearly in the center of the viewport. Ensure high contrast and steady lighting.")
            cam_img = st.camera_input("📷 Capture Verse")

        if cam_img:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("📸 Captured Frame")
                st.image(cam_img, use_container_width=True)

            with col2:
                st.subheader("Neural Processing...")
                with st.spinner("🔄 Deep OCR in progress..."):
                    result = call_upload_image(cam_img.getvalue(), "webcam_capture.jpg")

                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    ocr_text = result.get('ocr_extracted_text', '')
                    if ocr_text:
                        st.subheader("📝 Extracted Script")
                        st.markdown(f'<div class="ocr-box">{ocr_text}</div>', unsafe_allow_html=True)

                        edited = st.text_area("✏️ Refine Extracted Text:", value=ocr_text, height=140, key='web_edit')
                        if st.button("Finalize & Analyze", key='web_reanalyze'):
                            with st.spinner("Re-analyzing..."):
                                result = call_analyze(edited)

            if 'error' not in result:
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                show_results(result, expert)

    # ── TAB 2: UPLOAD ─────────────────────────────────────────────────────────
    with tab2:
        with st.container(border=True):
            uploaded = st.file_uploader("Upload Image (JPG, PNG, BMP)", type=["jpg","jpeg","png","bmp","tiff"])

        if uploaded:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("📸 Source Image")
                st.image(uploaded, use_container_width=True)

            with col2:
                st.subheader("Neural Processing...")
                with st.spinner("🔄 Running OCR engines..."):
                    result = call_upload_image(uploaded.getvalue(), uploaded.name)

                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    ocr_text = result.get('ocr_extracted_text', '')
                    if ocr_text:
                        st.subheader("📝 Extracted Script")
                        st.markdown(f'<div class="ocr-box">{ocr_text}</div>', unsafe_allow_html=True)

                        edited = st.text_area("✏️ Refine Extracted Text:", value=ocr_text, height=140, key='up_edit')
                        if st.button("Finalize & Analyze", key='up_reanalyze'):
                            with st.spinner("Re-analyzing..."):
                                result = call_analyze(edited)

            if 'error' not in result:
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                show_results(result, expert)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown("""
        <div class="hero">
            <h1>📊 Poetics Analytics Dashboard</h1>
            <p>High-fidelity structural & statistical insights across the poetic corpus</p>
        </div>
    """, unsafe_allow_html=True)
    
    df = load_genre_csv()
    if df is None:
        st.warning("No data found. Please run the pipeline or analyze some text first.")
        return

    # Premium Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="card"><h3>Total Corpus</h3><div class="val">{len(df):,}</div><div class="sub">Unique Poems</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><h3>Lexical Volume</h3><div class="val">{int(df["word_count"].mean())}</div><div class="sub">Avg Words/Poem</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><h3>Quality Index</h3><div class="val">{df["quality_score"].mean():.2f}</div><div class="sub">Avg Quality / 10</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="card"><h3>Figurative Logic</h3><div class="val">{df["figurative_primary"].nunique() if "figurative_primary" in df.columns else 0}</div><div class="sub">Distinct Modes</div></div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎭 Genre Distribution", "⭐ Quality Scores",
         "🎨 Figurative Language", "📐 Stylometrics"])

    with tab1:
        # Genre filter for the whole tab
        st.markdown("### 🎯 Distribution Filters")
        sel_genres = st.multiselect("View Genres:", 
                                   options=sorted(df['genre'].unique()),
                                   default=sorted(df['genre'].unique()),
                                   key='tab1_genres')
        
        df_tab1 = df[df['genre'].isin(sel_genres)].copy()
        counts = df_tab1['genre'].value_counts().reset_index()
        counts.columns = ['genre','count']
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig = px.bar(counts, x='genre', y='count', color='genre',
                         color_discrete_map=GENRE_COLOR, text='count',
                         title='Poems per Genre', template='plotly_dark')
            fig.update_traces(textposition='outside', marker_line_width=0)
            fig.update_layout(showlegend=False, height=450,
                              plot_bgcolor='rgba(0,0,0,0)', 
                              paper_bgcolor='rgba(0,0,0,0)',
                              font_family='Outfit',
                              title_font_size=20)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig2 = px.pie(counts, names='genre', values='count',
                          color='genre', color_discrete_map=GENRE_COLOR,
                          title='Genre Share', hole=0.5,
                          template='plotly_dark')
            fig2.update_layout(height=450,
                                plot_bgcolor='rgba(0,0,0,0)', 
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_family='Outfit')
            fig2.update_traces(textinfo='percent+label', pull=[0.05]*len(counts))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 🌳 Hierarchy & Relationship View")
        c1, c2 = st.columns(2)
        with c1:
            # Enhance Treemap
            fig_tree = px.treemap(counts, path=['genre'], values='count',
                                 color='genre', color_discrete_map=GENRE_COLOR,
                                 title='Genre interactive Hierarchy',
                                 template='plotly_dark')
            fig_tree.update_layout(height=400, font_family='Outfit')
            fig_tree.update_traces(textinfo="label+value")
            st.plotly_chart(fig_tree, use_container_width=True)
        
        with c2:
            # Sunburst (Genre -> Primary Figurative)
            if 'figurative_primary' in df.columns:
                st.markdown("#### Flow: Genre ➔ Figurative")
                # Option to hide 'none' for a better visual flow
                hide_none_sun = st.toggle("Filter 'none' (Sunburst)", value=True, key='sun_hide_none')
                df_sun = df_tab1.copy()
                if hide_none_sun:
                    df_sun = df_sun[df_sun['figurative_primary'] != 'none']
                
                fig_sun = px.sunburst(df_sun, path=['genre', 'figurative_primary'],
                                     color='genre', color_discrete_map=GENRE_COLOR,
                                     title='Interactive Stylometric Flow',
                                     template='plotly_dark')
                fig_sun.update_layout(height=450, font_family='Outfit',
                                      margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_sun, use_container_width=True)

    with tab2:
        if 'quality_score' in df.columns:
            st.markdown("### 🧬 Quality Distribution & Variance")
            fig = px.box(df, x='genre', y='quality_score', color='genre',
                         color_discrete_map=GENRE_COLOR, notched=True,
                         title='Quality Score Range by Genre',
                         template='plotly_dark')
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=60, b=20), font_family='Outfit')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🏹 Quality-Length Correlation")
            fig_scat = px.scatter(df, x='word_count', y='quality_score', color='genre',
                                  color_discrete_map=GENRE_COLOR, trendline="ols",
                                  opacity=0.6, marginal_x="histogram", marginal_y="box",
                                  title='Interactive Correlation Analysis',
                                  template='plotly_dark')
            fig_scat.update_layout(height=600, font_family='Outfit', clickmode='event+select')
            # Fix: scattergl unselected only supports marker and textfont
            fig_scat.update_traces(unselected=dict(marker=dict(opacity=0.05)))
            st.plotly_chart(fig_scat, use_container_width=True)

    with tab3:
        if 'figurative_primary' in df.columns:
            st.markdown("### 🎨 Figurative Language Analytics")
            ct = pd.crosstab(df['genre'], df['figurative_primary'], normalize='index')
            fig_heat = px.imshow(ct, text_auto=".1%", aspect="auto",
                                color_continuous_scale='Magma',
                                title='Usage Heatmap (By Genre %)',
                                template='plotly_dark')
            fig_heat.update_layout(height=500, font_family='Outfit')
            st.plotly_chart(fig_heat, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 🕸️ Figurative Signatures")
                ct_norm = ct.copy()
                for col in ct.columns: ct_norm[col] = ct[col] / (ct[col].max() if ct[col].max() > 0 else 1)
                m_fig = ct_norm.reset_index().melt(id_vars='genre')
                fig_spider = px.line_polar(m_fig, r='value', theta='figurative_primary', color='genre',
                                          line_close=True, color_discrete_map=GENRE_COLOR,
                                          title='Genre Figurative Fingerprint')
                fig_spider.update_layout(height=500, polar=dict(bgcolor='rgba(0,0,0,0)'), font_family='Outfit', template='plotly_dark')
                st.plotly_chart(fig_spider, use_container_width=True)
            
            with col_b:
                st.markdown("### 🍕 Type Distribution")
                fc = df['figurative_primary'].value_counts().reset_index()
                fc.columns = ['type','count']
                fig_pie = px.pie(fc, names='type', values='count', hole=0.5,
                              title='Overall Figurative Type Share', template='plotly_dark')
                fig_pie.update_layout(height=500, font_family='Outfit')
                st.plotly_chart(fig_pie, use_container_width=True)

            # Create a temporary numeric ID for coloring
            st.markdown("### ➰ Structural Flow")
            st.caption("How genres utilize figurative language. Ribbon density = Frequency.")
            
            # ── FILTERS for Flow ──
            col_fa, col_fb, col_fc = st.columns([2, 2, 1])
            with col_fa:
                flow_genres = st.multiselect("Filter Genres (Flow):", 
                                             options=sorted(df['genre'].unique()),
                                             default=sorted(df['genre'].unique()),
                                             key='flow_gen_filter')
            with col_fb:
                hide_none = st.toggle("Exclude 'None' types", value=True, help="Removes the 'none' figurative category for a cleaner flow")
            with col_fc:
                flow_sample = st.number_input("Sample:", 50, 5000, 1000, step=50, key='flow_sample')

            df_para = df[df['genre'].isin(flow_genres)].copy()
            if hide_none:
                df_para = df_para[df_para['figurative_primary'] != 'none']
            
            if len(df_para) > flow_sample:
                df_para = df_para.sample(flow_sample, random_state=42)

            genres_para = sorted(df_para['genre'].unique())
            genre_to_id = {g: i for i, g in enumerate(genres_para)}
            df_para['genre_id'] = df_para['genre'].map(genre_to_id)
            
            fig_para = px.parallel_categories(df_para, dimensions=['genre', 'figurative_primary'],
                                            labels={'genre': 'Genre', 'figurative_primary': 'Figurative Logic'},
                                            color='genre_id',
                                            color_continuous_scale='Viridis',
                                            title='Structural Alignment: Genre ➔ Logic Flow',
                                            template='plotly_dark')
            
            # Hide the numeric colorbar and use a manual legend or just rely on labels
            fig_para.update_layout(
                coloraxis_showscale=False,
                font_family='Outfit',
                margin=dict(l=80, r=120, t=80, b=40), # Increased margins significantly
                title_font_size=22
            )
            # Increase transparency of unselected paths for better interpretability
            fig_para.update_traces(hoveron='color', hoverinfo='count+probability',
                                  arrangement='freeform',
                                  bundlecolors=True)
            st.plotly_chart(fig_para, use_container_width=True)

    with tab4:
        # 1. On-the-fly Calculation of missing stylometrics
        def calc_stylo_cols(row):
            t = str(row['text'])
            w = re.findall(r'\b\w+\b', t)
            wc = len(w)
            if wc == 0: return pd.Series([0, 0, 0])
            awl = sum(len(x) for x in w) / wc
            lc = max(1, t.count('\n') + 1)
            # Rough syllable counting
            syl_count = sum(max(1, len(re.findall(r'[aeiouy]+', x.lower()))) for x in w)
            asw = syl_count / wc
            all_len = wc / lc
            return pd.Series([awl, asw, all_len])

        if 'avg_word_length' not in df.columns:
            df[['avg_word_length', 'avg_syllables_per_word', 'avg_line_length']] = df.apply(calc_stylo_cols, axis=1)

        num_cols = ['word_count', 'line_count', 'avg_word_length', 
                    'avg_syllables_per_word', 'avg_line_length']
        avail = [c for c in num_cols if c in df.columns]
        
        if avail:
            for c in avail:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            
            df = df[df['genre'].isin(GENRE_COLOR.keys())]
            grouped = df.groupby('genre')[avail].mean().reset_index()

            st.markdown("### 📊 Decoupled Stylometric Facts")
            st.caption("Factual comparisons per genre.")

            # Group 1: Scale (Word vs Line) - DECOUPLED INDIVIDUALS
            col1, col2 = st.columns(2)
            with col1:
                f1 = px.bar(grouped, x='genre', y='word_count', color='genre',
                            color_discrete_map=GENRE_COLOR, title='Avg Word Count',
                            template='plotly_dark')
                f1.update_layout(showlegend=False, height=350, font_family='Outfit')
                st.plotly_chart(f1, use_container_width=True)
            with col2:
                f2 = px.bar(grouped, x='genre', y='line_count', color='genre',
                            color_discrete_map=GENRE_COLOR, title='Avg Line Count',
                            template='plotly_dark')
                f2.update_layout(showlegend=False, height=350, font_family='Outfit')
                st.plotly_chart(f2, use_container_width=True)

            # Group 2: Complexity (Linguistic) - DECOUPLED INDIVIDUALS
            col3, col4 = st.columns(2)
            with col3:
                f3 = px.bar(grouped, x='genre', y='avg_word_length', color='genre',
                            color_discrete_map=GENRE_COLOR, title='Avg Word Length (Chars)',
                            template='plotly_dark')
                f3.update_layout(showlegend=False, height=350, font_family='Outfit')
                st.plotly_chart(f3, use_container_width=True)
            with col4:
                f4 = px.bar(grouped, x='genre', y='avg_syllables_per_word', color='genre',
                            color_discrete_map=GENRE_COLOR, title='Avg Syllables/Word',
                            template='plotly_dark')
                f4.update_layout(showlegend=False, height=350, font_family='Outfit')
                st.plotly_chart(f4, use_container_width=True)

            # REFINED: Factual Comparison - Feature Variance
            st.markdown("### 📦 Feature Distribution Comparison")
            st.caption("Each feature separated for absolute scale readability.")
            
            for col in avail:
                with st.expander(f"View Distribution: {col.replace('_', ' ').title()}", expanded=(col == 'word_count')):
                    fig_box = px.box(df, x='genre', y=col, color='genre',
                                    color_discrete_map=GENRE_COLOR,
                                    points="outliers",
                                    title=f'Variance: {col.replace("_", " ").title()}',
                                    template='plotly_dark')
                    fig_box.update_layout(height=400, font_family='Outfit',
                                         plot_bgcolor='rgba(0,0,0,0)', 
                                         paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_box, use_container_width=True)

            # NEW: Parallel Coordinates (Scientific Multi-axis Flow)
            st.markdown("### 🪜 Parallel Feature Coordinate Flow")
            st.caption("How specific genres 'flow' across metrics. Use filters to reduce clutter.")
            
            # ── FILTERS ──
            col_f1, col_f2 = st.columns([2, 1])
            with col_f1:
                selected_genres = st.multiselect("Filter Genres:", 
                                                 options=sorted(df['genre'].unique()),
                                                 default=sorted(df['genre'].unique())[:3])
            with col_f2:
                sample_n = st.slider("Sample Size:", 100, 1000, 300)

            pc_df = df[df['genre'].isin(selected_genres)].copy()
            if len(pc_df) > sample_n:
                pc_df = pc_df.sample(sample_n, random_state=42)
            
            # Map genres to IDs for the color scale
            genres_in_view = sorted(pc_df['genre'].unique())
            genre_to_id = {g: i for i, g in enumerate(genres_in_view)}
            pc_df['genre_id'] = pc_df['genre'].map(genre_to_id)
            
            # Use go.Parcoords for better control over axis labels
            # Start with Genre as the FIRST axis for clarity
            dimensions = [dict(
                range=[0, len(genres_in_view) - 1],
                tickvals=list(range(len(genres_in_view))),
                ticktext=genres_in_view,
                label='GENRE (Target)',
                values=pc_df['genre_id']
            )]
            
            for col in avail:
                dimensions.append(dict(
                    range=[pc_df[col].min(), pc_df[col].max()],
                    label=col.replace('_', ' ').title(),
                    values=pc_df[col]
                ))

            fig_pc = go.Figure(data=go.Parcoords(
                line=dict(color=pc_df['genre_id'],
                          colorscale='Viridis',
                          showscale=True,
                          colorbar=dict(
                              title='Genre legend',
                              tickvals=list(range(len(genres_in_view))),
                              ticktext=genres_in_view,
                              x=1.12
                          )),
                dimensions=dimensions,
                labelfont=dict(size=14, family='Outfit', color='white'),
                tickfont=dict(size=12, family='Outfit', color='rgba(255,255,255,0.7)'),
                unselected=dict(line=dict(color='rgba(100,100,100,0.02)', opacity=0.01))
            ))
            
            fig_pc.update_layout(
                title=f'Structural DNA ({len(pc_df)} poems sampled)',
                template='plotly_dark',
                height=600,
                font_family='Outfit'
            )
            st.plotly_chart(fig_pc, use_container_width=True)

            # NEW: SPLOM (Scatter Plot Matrix)
            st.markdown("### 🔲 Scatter Matrix (SPLOM)")
            st.caption("Correlation grid mapping all features against each other.")
            fig_splom = px.scatter_matrix(df, dimensions=avail, color='genre',
                                         color_discrete_map=GENRE_COLOR,
                                         opacity=0.3,
                                         title='Feature Correlation Grid',
                                         template='plotly_dark')
            fig_splom.update_layout(height=800, font_family='Outfit')
            fig_splom.update_traces(diagonal_visible=False, marker=dict(size=3))
            st.plotly_chart(fig_splom, use_container_width=True)

            # Radar Profiler
            st.markdown("### 🕸️ Profiler: Stylometric Fingerprints")
            st.caption("Relative shape comparison normalized for a cohesive visual signature.")
            
            grouped_norm = grouped.copy()
            for c in avail:
                max_val = grouped[c].max()
                if max_val > 0:
                    grouped_norm[c] = grouped[c] / max_val
            
            m_norm = grouped_norm.melt(id_vars='genre', value_vars=avail)
            fig_radar = px.line_polar(m_norm, r='value', theta='variable', color='genre',
                                     line_close=True, template='plotly_dark',
                                     color_discrete_map=GENRE_COLOR,
                                     title='Multi-Genre Stylometric Signature')
            fig_radar.update_layout(height=550, polar=dict(bgcolor='rgba(0,0,0,0)'),
                                    font_family='Outfit')
            st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def page_model_results():
    st.markdown("""<div class="header">
        <h1>🏆 Model Results</h1>
        <p>Training and evaluation metrics for all ML models</p>
    </div>""", unsafe_allow_html=True)

    results = None
    res_path = os.path.join(SAVED_MODELS, 'training_results.json')
    if os.path.exists(res_path):
        import json
        with open(res_path, 'r') as f:
            results = json.load(f)

    if not results:
        st.warning("No results found. Run: `python models/train_models.py`")
        return

    for task, task_res in results.items():
        st.subheader(f"📊 {task.replace('_',' ').title()}")
        rows = []
        for model, m in task_res.items():
            row = {'Model': model}
            row.update({k: round(v,4) if isinstance(v,float) else v
                        for k,v in m.items() if k != 'confusion_matrix'})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("---")




# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Monolithic mode - No backend startup needed

    pages = {
        "🏠 Home":           page_home,
        "🔍 Analyze Text":   None,
        "📷 Camera & Image": None,
        "📊 Dashboard":      page_dashboard,
        "🏆 Model Results":  page_model_results,
    }

    with st.sidebar:
        st.markdown("""<div style="text-align:center;padding:1rem 0 .5rem">
            <div style="font-size:2.8rem">📜</div>
            <div style="font-weight:bold;font-size:1.1rem;color:white">Computational</div>
            <div style="color:#aad4f5;font-size:1.1rem">Poetics Analyzer</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        default_page = st.session_state.pop('goto', "🏠 Home")
        try:
            default_idx = list(pages.keys()).index(default_page)
        except ValueError:
            default_idx = 0
        page = st.radio("", list(pages.keys()), index=default_idx)

        st.markdown("---")
        # Expert mode removed per user request
        st.markdown("🌐 **Engine: Monolithic (Local)**")
        st.caption("AI models loaded directly in-app.")
        
        # Simple model load check
        n_loaded = sum(1 for m in [predictor.genre_model, predictor.fig_model, 
                                   predictor.emo_model, predictor.qual_model] if m)
        st.markdown(f"<small>✅ {n_loaded}/4 Models Ready</small>", unsafe_allow_html=True)

        st.markdown("---")
        st.caption("v3.0 · 4 Models · Native OCR")
        try:
            from ocr.image_to_text import OCR_CONFIG
            if not OCR_CONFIG['EASY_OK']:
                st.info("💡 **OCR Tip**: Native OCR is disabled (library missing).")
        except: pass

    # Route
    if   page == "🔍 Analyze Text":   page_analyze(False)
    elif page == "📷 Camera & Image": page_camera(False)
    elif pages[page]:                  pages[page]()


if __name__ == "__main__":
    main()
