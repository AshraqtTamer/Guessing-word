import streamlit as st
import random
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer  # Swapped from gensim

# =====================================================================
# 🧠 LIGHTWEIGHT BACKEND ENGINE & CACHING
# =====================================================================

@st.cache_resource
def load_semantic_model():
    # Only 90MB - Boots instantly and runs perfectly on free servers!
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def build_word2vec_dataset():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    topics = []
    descriptions = []
    target_categories = ['animal', 'food', 'plant', 'flower', 'artifact']
    
    for synset in list(wn.all_synsets(pos='n')):
        lex_name = synset.lexname()
        if any(cat in lex_name for cat in target_categories):
            word = synset.lemmas()[0].name().replace('_', ' ')
            definition = synset.definition()
            
            if 3 < len(word) < 12 and len(definition) > 25 and word.isalpha():
                topics.append(word.capitalize())
                descriptions.append(definition)
                
    return pd.DataFrame({"Topic": topics, "Description": descriptions})

with st.spinner("🔄 Loading Lightweight Transformer Model (90MB)..."):
    model = load_semantic_model()

with st.spinner("🔄 Building Dataset from WordNet..."):
    df = build_word2vec_dataset()

# =====================================================================
# ⚙️ STATE MANAGEMENT (Backend Session Logic)
# =====================================================================

def get_valid_secret_word():
    idx = random.randint(0, len(df) - 1)
    word = df.iloc[idx]['Topic']
    desc = df.iloc[idx]['Description']
    return word, desc

if 'secret_word' not in st.session_state:
    word, desc = get_valid_secret_word()
    st.session_state.secret_word = word
    st.session_state.secret_desc = desc
    st.session_state.guess_history = {}
    st.session_state.game_over = False

def reset_game():
    word, desc = get_valid_secret_word()
    st.session_state.secret_word = word
    st.session_state.secret_desc = desc
    st.session_state.guess_history = {}
    st.session_state.game_over = False

# Updated Similarity Vector Engine
def calculate_w2v_score(guess, secret):
    guess_clean = guess.strip().lower()
    secret_clean = secret.strip().lower()
    
    # SentenceTransformers vectorizes sentences/words out-of-the-box
    vec1 = model.encode([guess_clean])
    vec2 = model.encode([secret_clean])
    
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return round(max(0, similarity) * 100, 2)

# =====================================================================
# 🎨 FRONTEND UI DESIGN (Streamlit Layout)
# =====================================================================

st.set_page_config(page_title="Semantic Clueless", page_icon="🚀", layout="centered")

st.title("🚀 Semantic Clueless (Transformer Edition)")
st.write("Read the description provided by the AI and try to guess the exact single-word item.")

st.markdown("---")

st.subheader("📖 AI Clue Description:")
st.info(f'"{st.session_state.secret_desc}"')

with st.form(key='guess_form', clear_on_submit=True):
    user_input = st.text_input("🤔 Your Guess Word:", disabled=st.session_state.game_over).strip()
    submit_button = st.form_submit_button(label='Submit Guess', disabled=st.session_state.game_over)

if submit_button and user_input:
    guess_lower = user_input.lower()
    secret_lower = st.session_state.secret_word.lower()
    
    if guess_lower == secret_lower:
        st.balloons()
        st.success(f"🎯 **CORRECT!** The model matched vectors perfectly! The word is: **{st.session_state.secret_word}**")
        st.session_state.game_over = True
    else:
        score_pct = calculate_w2v_score(user_input, st.session_state.secret_word)
        st.session_state.guess_history[user_input] = score_pct
        
        if score_pct > 65:
            st.error(f"🔥 **Extremely Hot!** '{user_input}' is very close! ({score_pct}%)")
        elif score_pct > 35:
            st.warning(f"🌤️ **Getting Warmer!** '{user_input}' shares some context. ({score_pct}%)")
        else:
            st.info(f"❄️ **Freezing Cold!** '{user_input}' is far off. ({score_pct}%)")

col1, col2 = st.columns([4, 1])
with col1:
    if st.button("🔄 Next Word / Skip"):
        reset_game()
        st.rerun()
with col2:
    if st.button("🏳️ Give Up"):
        st.write(f"The secret word was: **{st.session_state.secret_word}**")
        st.session_state.game_over = True

if st.session_state.guess_history:
    st.sidebar.title("📊 Guess Proximity")
    sorted_history = sorted(st.session_state.guess_history.items(), key=lambda x: x[1], reverse=True)
    
    for word, score in sorted_history:
        if score > 65:
            st.sidebar.markdown(f"🔴 **{word}** → `{score}%` (Hot)")
        elif score > 35:
            st.sidebar.markdown(f"🟡 **{word}** → `{score}%` (Warm)")
        else:
            st.sidebar.markdown(f"🔵 {word} → `{score}%` (Cold)")
