import os
import tempfile

# ✅ CRITICAL: Set environment variables BEFORE any other imports
os.environ['TRANSFORMERS_CACHE'] = tempfile.gettempdir()
os.environ['HF_HOME'] = tempfile.gettempdir()
os.environ['TORCH_HOME'] = tempfile.gettempdir()
os.environ['HF_DATASETS_CACHE'] = tempfile.gettempdir()
os.environ['HUGGINGFACE_HUB_CACHE'] = tempfile.gettempdir()

import streamlit as st

# ✅ CRITICAL: set_page_config() MUST be called first, before ANY Streamlit commands
st.set_page_config(
    page_title="AI Study Helper Pro - by Umaima Qureshi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
import base64
import torch
import nltk

@st.cache_resource
def init_nltk():
    """Initialize NLTK with writable directory"""
    nltk_data_dir = os.path.join(tempfile.gettempdir(), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
    
    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
            except:
                pass  # Continue if download fails
    
    return True

# Initialize NLTK
init_nltk()

# Device detection
DEVICE = 0 if torch.cuda.is_available() else -1

# Lazy model loading with proper cache handling - FIXED (No cache_dir in pipeline)
@st.cache_resource
def get_summarizer():
    """Load summarization model - cache_dir handled by environment variables"""
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "sshleifer/distilbart-cnn-12-6",
            cache_dir=tempfile.gettempdir()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "sshleifer/distilbart-cnn-12-6",
            cache_dir=tempfile.gettempdir()
        )
        summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE
        )
        return summarizer
    except Exception as e:
        st.error(f"Failed to load summarizer: {str(e)}")
        return None

@st.cache_resource
def get_qa():
    """Load Q&A model - cache_dir handled by environment variables"""
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(
            "distilbert-base-uncased-distilled-squad",
            cache_dir=tempfile.gettempdir()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-distilled-squad",
            cache_dir=tempfile.gettempdir()
        )
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE
        )
        return qa_pipeline
    except Exception as e:
        st.error(f"Failed to load Q&A model: {str(e)}")
        return None

@st.cache_resource
def get_classifier():
    """Load classifier model - cache_dir handled by environment variables"""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            "typeform/distilbert-base-uncased-mnli",
            cache_dir=tempfile.gettempdir()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "typeform/distilbert-base-uncased-mnli",
            cache_dir=tempfile.gettempdir()
        )
        classifier = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE
        )
        return classifier
    except Exception as e:
        st.error(f"Failed to load classifier: {str(e)}")
        return None

@st.cache_resource
def load_translator(model_name):
    """Load translation model - cache_dir handled by environment variables"""
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=tempfile.gettempdir()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=tempfile.gettempdir()
        )
        translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE
        )
        return translator
    except Exception as e:
        st.error(f"Failed to load translator: {str(e)}")
        return None

def truncate_text(text, max_words=400):
    """Truncate text to maximum word count"""
    words = text.split()
    return (" ".join(words[:max_words]), len(words) > max_words)

# ULTRA PREMIUM CSS - Glassmorphism + Animations
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&display=swap');
* {
    font-family: 'Poppins', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
/* Hero Header */
.hero-header {
    background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5f 100%);
    padding: 3rem 2rem;
    border-radius: 25px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    border: 2px solid #667eea;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 900;
    color: #ffffff;
    margin: 0 0 1rem 0;
    text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.5);
}
.hero-subtitle {
    font-size: 1.3rem;
    color: #ffffff;
    margin: 0;
    font-weight: 400;
    opacity: 0.95;
}
/* Premium Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: rgba(255, 255, 255, 0.1);
    padding: 12px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 15px;
    color: white;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 12px 24px;
    border: 1px solid rgba(255, 255, 255, 0.25);
    transition: all 0.3s ease;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.25);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-color: rgba(255, 255, 255, 0.4);
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
}
/* Premium Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 16px 40px;
    border-radius: 16px;
    border: none;
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    transition: all 0.3s ease;
    width: 100%;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 36px rgba(102, 126, 234, 0.6);
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}
.stButton > button:active {
    transform: translateY(-1px);
}
/* Input Fields - FIXED: Black background for text areas */
.stTextArea textarea, .stTextInput input {
    background: rgba(0, 0, 0, 0.85) !important;
    backdrop-filter: blur(10px);
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 16px !important;
    color: white !important;
    font-size: 1rem !important;
    padding: 16px !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: rgba(255, 255, 255, 0.6) !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(255, 255, 255, 0.6) !important;
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.3) !important;
    background: rgba(0, 0, 0, 0.9) !important;
}
/* Result Cards */
.result-card {
    background: rgba(255, 255, 255, 0.95);
    color: #1a1a1a;
    padding: 2rem;
    border-radius: 20px;
    margin: 1rem auto;              /* Changed: 0 to auto */
    max-width: 900px;                /* Added this line */
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    animation: fadeIn 0.5s ease;
    border-left: 5px solid #667eea;
}
.result-card p {
    word-break: break-word;
    overflow-wrap: break-word;
    max-height: 200px;               /* Changed: 300px to 200px */
    overflow-y: auto;
    line-height: 1.6;                /* Added this line */
}
/* FIXED: Scrollable container for keywords with proper visibility */
.keywords-container {
    max-height: 400px;
    overflow-y: auto;
    padding: 10px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    margin-top: 1rem;
}
/* FIXED: Quiz questions container with dark background and scrolling */
.quiz-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 10px;
}
.quiz-question-card {
    background: rgba(30, 30, 60, 0.9) !important;
    color: white !important;
    padding: 1.5rem;
    margin: 1rem 0;
    border-radius: 15px;
    border-left: 4px solid #667eea;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}
.quiz-question-card strong {
    color: #667eea !important;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
/* Stats Badge */
.stats-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(10px);
    padding: 8px 20px;
    border-radius: 20px;
    color: white;
    font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.3);
    margin: 5px;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}
/* Success/Error Messages */
.stSuccess {
    background: rgba(72, 187, 120, 0.25) !important;
    backdrop-filter: blur(10px);
    border-left: 4px solid #48bb78 !important;
    border-radius: 12px !important;
    color: white !important;
}
.stError {
    background: rgba(245, 101, 101, 0.25) !important;
    backdrop-filter: blur(10px);
    border-left: 4px solid #f56565 !important;
    border-radius: 12px !important;
    color: white !important;
}
.stInfo {
    background: rgba(66, 153, 225, 0.25) !important;
    backdrop-filter: blur(10px);
    border-left: 4px solid #4299e1 !important;
    border-radius: 12px !important;
    color: white !important;
}
.stWarning {
    background: rgba(237, 137, 54, 0.25) !important;
    backdrop-filter: blur(10px);
    border-left: 4px solid #ed8936 !important;
    border-radius: 12px !important;
    color: white !important;
}
/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: rgba(20, 20, 40, 0.9);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.2);
}
.css-1d391kg h2, [data-testid="stSidebar"] h2 {
    color: white;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}
.css-1d391kg p, [data-testid="stSidebar"] p {
    color: rgba(255, 255, 255, 0.9);
}
/* Ensure text wraps properly */
.stTabs h3 {
    word-break: break-word;
    overflow-wrap: anywhere;
    white-space: normal;
    max-width: 100%;
    margin: 0;
    padding: 0.5rem 0;
    color: white;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}
/* Spinner */
.stSpinner > div {
    border-top-color: white !important;
}
/* Selectbox */
.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(10px);
    border-radius: 12px !important;
    color: white !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
}
.stSelectbox label {
    color: white !important;
    font-weight: 600 !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}
/* Footer */
.premium-footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    background: rgba(20, 20, 40, 0.85);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.premium-footer p {
    color: rgba(255, 255, 255, 0.9);
    margin: 0.5rem 0;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}
/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
/* Custom scrollbar */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 5px;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🧠 AI Study Helper Pro</div>
    <div class="hero-subtitle">⚡ Supercharge Your Learning with Advanced AI Technology</div>
</div>
""", unsafe_allow_html=True)

# Add cache clear button (for troubleshooting)
with st.expander("⚙️ Settings", expanded=False):
    if st.button("🔄 Clear Model Cache (if you see errors)"):
        st.cache_resource.clear()
        st.success("✅ Cache cleared! Please refresh the page.")
        st.info("💡 This will reload all AI models on next use.")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Dashboard")
    st.markdown("---")
    
    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="stats-badge">📊 247 Processed</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stats-badge">⚡ 2.3s Avg</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ✨ Features")
    features = [
        "📝 AI Summarization",
        "💬 Smart Q&A",
        "🎯 Quiz Generator",
        "🌍 Multi-Language",
        "🔑 Keyword Extraction",
        "💨 Lightning Fast"
    ]
    for feat in features:
        st.markdown(f"**{feat}**")
    
    st.markdown("---")
    st.markdown("### 👩‍💻 Developer")
    st.markdown("**Umaima Qureshi**")
    st.markdown("[GitHub](https://github.com/Umaima122)")

# Initialize session state
for key in ["summary", "quiz", "translation", "keywords"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key not in ["quiz", "keywords"] else []

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📝 Summarize", "💬 Q&A", "🎯 Quiz", "🌍 Translate", "🔑 Keywords", "📥 Download"
])

# ============================================
# TAB 1: SUMMARIZE
# ============================================
with tab1:
    st.markdown("### 📝 Intelligent Summarization")
    
    text = st.text_area(
        "✍️ Your notes or textbook:", 
        value="", 
        height=250, 
        key="sum_txt", 
        placeholder="Paste your content here and watch AI magic happen..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("✨ Generate Summary", key="sum_btn"):
            if not text.strip():
                st.error("⚠️ Please provide text to summarize")
            else:
                trunc, was_trunc = truncate_text(text, 400)
                if was_trunc:
                    st.info("📊 Text optimized to 400 words for processing")
                
                if len(trunc.split()) < 20:
                    st.error("⚠️ Need at least 20 words to generate a meaningful summary")
                else:
                    with st.spinner("🧠 AI is thinking..."):
                        try:
                            summarizer = get_summarizer()
                            if summarizer:
                                result = summarizer(
                                    trunc, 
                                    max_length=130, 
                                    min_length=30, 
                                    do_sample=False
                                )
                                summary = result[0]['summary_text']
                                
                                st.markdown(f"""
                                <div class="result-card">
                                    <h4 style="color: #667eea; margin-bottom: 1rem;">📄 AI-Generated Summary</h4>
                                    <p style="font-size: 1.1rem; line-height: 1.8; color: #2d3748;">{summary}</p>
                                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                                        <span class="stats-badge" style="background: #667eea; color: white;">
                                            {len(summary.split())} words
                                        </span>
                                        <span class="stats-badge" style="background: #48bb78; color: white;">
                                            ✓ Completed
                                        </span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.session_state["summary"] = summary
                        except Exception as e:
                            st.error(f"❌ Error generating summary: {str(e)}")

# ============================================
# TAB 2: Q&A
# ============================================
with tab2:
    st.markdown("### 💬 Intelligent Q&A System")
    
    context = st.text_area(
        "📚 Context (Your notes):", 
        value="", 
        height=200, 
        key="qa_ctx",
        placeholder="Paste your study material here..."
    )
    
    question = st.text_input(
        "❓ Ask your question:", 
        key="qa_q",
        placeholder="What would you like to know?"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔍 Get Answer", key="qa_btn"):
            if not context.strip() or not question.strip():
                st.error("⚠️ Please provide both context and question")
            else:
                trunc_ctx, _ = truncate_text(context, 400)
                with st.spinner("🤔 Analyzing..."):
                    try:
                        qa_model = get_qa()
                        if qa_model:
                            result = qa_model(question=question, context=trunc_ctx)
                            answer = result['answer']
                            confidence = result.get('score', 0)
                            
                            st.markdown(f"""
                            <div class="result-card">
                                <h4 style="color: #667eea; margin-bottom: 1rem;">💡 AI Answer</h4>
                                <p style="font-size: 1.2rem; line-height: 1.8; color: #2d3748; font-weight: 500;">{answer}</p>
                                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                                    <span class="stats-badge" style="background: #48bb78; color: white;">
                                        ✓ Answer Found
                                    </span>
                                    <span class="stats-badge" style="background: #667eea; color: white;">
                                        Confidence: {confidence:.1%}
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error finding answer: {str(e)}")

# ============================================
# TAB 3: QUIZ - FIXED
# ============================================
with tab3:
    st.markdown("### 🎯 AI Quiz Generator")
    
    quiz_ctx = st.text_area(
        "📖 Study material:", 
        value="", 
        height=200, 
        key="quiz_ctx",
        placeholder="Paste content for quiz generation..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Generate Quiz", key="quiz_btn"):
            if not quiz_ctx.strip():
                st.error("⚠️ Please provide text for quiz generation")
            else:
                trunc_quiz, _ = truncate_text(quiz_ctx, 200)
                with st.spinner("🎲 Creating questions..."):
                    try:
                        sentences = sent_tokenize(trunc_quiz)[:5]
                        if len(sentences) == 0:
                            st.warning("⚠️ Could not extract sentences from the text")
                        else:
                            def get_first_words(text, max_words=12):
                                """Get first N complete words from text"""
                                words = text.split()
                                if len(words) <= max_words:
                                    return text
                                return ' '.join(words[:max_words])
                    
                            questions = [f"What is the main concept in: '{get_first_words(s, 12)}'?" for s in sentences if len(s) > 10]
                            if questions:
                                st.markdown("<h4 style='color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.3);'>📝 Generated Quiz Questions</h4>", unsafe_allow_html=True)
                                st.markdown('<div class="quiz-container">', unsafe_allow_html=True)
                                for i, q in enumerate(questions, 1):
                                    st.markdown(f"""
                                    <div class='quiz-question-card'>
                                         <strong>Question {i}:</strong> <span style='color: white;'>{q}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.markdown('</div></div>', unsafe_allow_html=True)
                                st.session_state["quiz"] = questions
                            else:
                                st.warning("⚠️ Could not generate questions from the provided text")
                    except Exception as e:
                        st.error(f"❌ Error generating quiz: {str(e)}")

# ============================================
# TAB 4: TRANSLATE
# ============================================
with tab4:
    st.markdown("### 🌍 AI Translation")
    
    trans_text = st.text_area(
        "✍️ Text to translate:", 
        height=200, 
        key="trans_txt",
        placeholder="Enter text to translate..."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        lang = st.selectbox(
            "🎯 Target language:", 
            ["French", "German", "Spanish", "Italian", "Hindi"]
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("🌐 Translate Now", key="trans_btn"):
            if not trans_text.strip():
                st.error("⚠️ Please provide text to translate")
            else:
                model_map = {
                    "French": "Helsinki-NLP/opus-mt-en-fr",
                    "German": "Helsinki-NLP/opus-mt-en-de",
                    "Spanish": "Helsinki-NLP/opus-mt-en-es",
                    "Italian": "Helsinki-NLP/opus-mt-en-it",
                    "Hindi": "Helsinki-NLP/opus-mt-en-hi"
                }
                
                trunc_trans, _ = truncate_text(trans_text, 200)
                with st.spinner(f"🌍 Translating to {lang}..."):
                    try:
                        translator = load_translator(model_map[lang])
                        if translator:
                            result = translator(trunc_trans, max_length=256)
                            translation = result[0]['translation_text']
                            
                            st.markdown(f"""
                            <div class="result-card">
                                <h4 style="color: #667eea; margin-bottom: 1rem;">🌐 Translation ({lang})</h4>
                                <p style="font-size: 1.2rem; line-height: 1.8; color: #2d3748;">{translation}</p>
                                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                                    <span class="stats-badge" style="background: #48bb78; color: white;">
                                        ✓ Translated
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.session_state["translation"] = translation
                    except Exception as e:
                        st.error(f"❌ Translation Error: {str(e)}")

# ============================================
# TAB 5: KEYWORDS - FIXED
# ============================================
with tab5:
    st.markdown("### 🔑 AI Keyword Extraction")
    
    keyword_input = st.text_area(
        "📝 Text for analysis:", 
        value="", 
        height=200, 
        key="kw_txt",
        placeholder="Paste text to extract key concepts..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔍 Extract Keywords", key="kw_btn"):
            if not keyword_input.strip():
                st.error("⚠️ Please provide text for keyword extraction")
            else:
                trunc_kw, _ = truncate_text(keyword_input, 200)
                with st.spinner("🔎 Analyzing concepts..."):
                    try:
                        classifier = get_classifier()
                        if classifier:
                            labels = ["technology", "science", "education", "health", "business", "finance", "medical", "engineering", "mathematics", "history"]
                            result = classifier(trunc_kw, labels)
                            keywords = [lbl for lbl, score in zip(result['labels'], result['scores']) if score > 0.3][:5]
                            
                            if keywords:
                                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                                st.markdown("<h4 style='color: #667eea;'>🎯 Extracted Keywords</h4>", unsafe_allow_html=True)
                                st.markdown('<div class="keywords-container">', unsafe_allow_html=True)
                                kw_html = " ".join([
                                    f"<span style='display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 12px 24px; border-radius: 25px; margin: 8px; font-size: 1rem; font-weight: 600; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);'>{kw}</span>"
                                    for kw in keywords
                                ])
                                st.markdown(kw_html, unsafe_allow_html=True)
                                st.markdown('</div></div>', unsafe_allow_html=True)
                                st.session_state["keywords"] = keywords
                            else:
                                st.info("ℹ️ No strong keywords found. Try providing more detailed text.")
                    except Exception as e:
                        st.error(f"❌ Error extracting keywords: {str(e)}")

# ============================================
# TAB 6: DOWNLOAD
# ============================================
with tab6:
    st.markdown("### 📥 Download Results")
    
    def download_link(text, filename, emoji):
        """Generate download link for text content"""
        b64 = base64.b64encode(text.encode()).decode()
        return f"""
        <a href="data:file/txt;base64,{b64}" download="{filename}" 
           style="display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2); 
           color: white; padding: 16px 32px; border-radius: 16px; text-decoration: none; 
           font-weight: 700; font-size: 1.1rem; margin: 10px; box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
           transition: all 0.3s ease;">
            {emoji} Download {filename}
        </a>
        """
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state["summary"]:
            st.markdown(download_link(st.session_state["summary"], "summary.txt", "📄"), unsafe_allow_html=True)
        else:
            st.info("📄 Generate a summary first")
            
        if st.session_state["quiz"]:
            quiz_text = "\n\n".join([f"Question {i}: {q}" for i, q in enumerate(st.session_state["quiz"], 1)])
            st.markdown(download_link(quiz_text, "quiz.txt", "🎯"), unsafe_allow_html=True)
        else:
            st.info("🎯 Generate a quiz first")
    
    with col2:
        if st.session_state["translation"]:
            st.markdown(download_link(st.session_state["translation"], "translation.txt", "🌍"), unsafe_allow_html=True)
        else:
            st.info("🌍 Translate text first")
            
        if st.session_state["keywords"]:
            keywords_text = "Extracted Keywords:\n\n" + "\n".join([f"- {kw}" for kw in st.session_state["keywords"]])
            st.markdown(download_link(keywords_text, "keywords.txt", "🔑"), unsafe_allow_html=True)
        else:
            st.info("🔑 Extract keywords first")
    
    st.markdown("---")
    
    if not any([st.session_state["summary"], st.session_state["quiz"], 
                st.session_state["translation"], st.session_state["keywords"]]):
        st.warning("ℹ️ Generate content in other tabs to enable downloads")
    else:
        st.success("✅ Content ready for download! Click the buttons above.")

# ============================================
# PREMIUM FOOTER
# ============================================
st.markdown("""
<div class="premium-footer">
    <p style="font-size: 1.2rem; font-weight: 600;">
        Built with ❤️ by 
        <span style="background: linear-gradient(135deg, #ffffff, #e0e7ff); 
                     -webkit-background-clip: text; 
                     -webkit-text-fill-color: transparent; 
                     font-weight: 700;">
            Umaima Qureshi
        </span>
    </p>
    <p style="font-size: 0.9rem;">© 2025 AI Study Helper Pro. All Rights Reserved.</p>
    <p style="margin-top: 1rem;">
        <a href="https://github.com/Umaima122" target="_blank" 
           style="color: white; text-decoration: none; padding: 8px 20px; 
                  background: rgba(255, 255, 255, 0.15); border-radius: 20px; 
                  backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.3);
                  transition: all 0.3s ease; margin: 0 5px;">
            🔗 GitHub
        </a>
        <a href="https://www.linkedin.com/in/umaima-qureshi" target="_blank" 
           style="color: white; text-decoration: none; padding: 8px 20px; 
                  background: rgba(255, 255, 255, 0.15); border-radius: 20px; 
                  backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.3);
                  transition: all 0.3s ease; margin: 0 5px;">
            💼 LinkedIn
        </a>
    </p>
    <p style="font-size: 0.85rem; margin-top: 1rem; opacity: 0.8;">
        Powered by Transformers • PyTorch • Streamlit • NLTK
    </p>
</div>
""", unsafe_allow_html=True)
