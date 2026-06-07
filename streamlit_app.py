
import streamlit as st
import random
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

# =====================================================================
# 🧠 BACKEND ENGINE & CACHING (Loads once into Server Memory)
# =====================================================================

@st.cache_resource # Caches the 1.6 GB model in server RAM so it never reloads on clicks
def load_word2vec_model():
    return api.load('word2vec-google-news-300')

@st.cache_data # Caches the data collection to save processing time
def build_word2vec_dataset():
    nltk.download('wordnet', quiet=True)
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

# Loading indicators to inform the user during initialization
with st.spinner("🔄 Loading Word2Vec Google News Model (1.6 GB)... This might take a minute on first boot."):
    w2v_model = load_word2vec_model()

with st.spinner("🔄 Building Dataset from WordNet..."):
    df = build_word2vec_dataset()

# =====================================================================
# ⚙️ STATE MANAGEMENT (Backend Session Logic)
# =====================================================================

def get_valid_secret_word():
    """Selects a random word and guarantees it exists within Google's Word2Vec dict"""
    while True:
        idx = random.randint(0, len(df) - 1)
        word = df.iloc[idx]['Topic']
        desc = df.iloc[idx]['Description']
        if word.lower() in w2v_model:
            return word, desc

# Initialize session components if starting fresh
if 'secret_word' not in st.session_state:
    word, desc = get_valid_secret_word()
    st.session_state.secret_word = word
    st.session_state.secret_desc = desc
    st.session_state.guess_history = {}
    st.session_state.game_over = False

def reset_game():
    """Resets states to spawn a new game round"""
    word, desc = get_valid_secret_word()
    st.session_state.secret_word = word
    st.session_state.secret_desc = desc
    st.session_state.guess_history = {}
    st.session_state.game_over = False

# Your core mathematical similarity logic
def calculate_w2v_score(guess, secret):
    guess_clean = guess.strip().lower()
    secret_clean = secret.strip().lower()
    
    if guess_clean not in w2v_model or secret_clean not in w2v_model:
        return None  # Out Of Vocabulary flag
        
    vec1 = w2v_model[guess_clean].reshape(1, -1)
    vec2 = w2v_model[secret_clean].reshape(1, -1)
    
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return round(max(0, similarity) * 100, 2)

# =====================================================================
# 🎨 FRONTEND UI DESIGN (Streamlit Layout)
# =====================================================================

st.set_page_config(page_title="Word2Vec Semantic Clueless", page_icon="🚀", layout="centered")

st.title("🚀 Semantic Clueless (Word2Vec Edition)")
st.write("Read the description provided by the AI and try to guess the exact single-word item. The AI computes the semantic proximity using pre-trained Google News Embeddings!")

st.markdown("---")

# Display Active Riddle Clue
st.subheader("📖 AI Clue Description:")
st.info(f'"{st.session_state.secret_desc}"')

# User Interaction Form
with st.form(key='guess_form', clear_on_submit=True):
    user_input = st.text_input("🤔 Your Guess Word:", disabled=st.session_state.game_over).strip()
    submit_button = st.form_submit_button(label='Submit Guess', disabled=st.session_state.game_over)

# Backend evaluation triggered on form submission
if submit_button and user_input:
    guess_lower = user_input.lower()
    secret_lower = st.session_state.secret_word.lower()
    
    if guess_lower == secret_lower:
        st.balloons()
        st.success(f"🎯 **CORRECT!** Word2Vec matched vectors perfectly! The word is: **{st.session_state.secret_word}**")
        st.session_state.game_over = True
    else:
        score_pct = calculate_w2v_score(user_input, st.session_state.secret_word)
        
        if score_pct is None:
            st.warning(f"⚠️ '{user_input}' is not recognized in the Word2Vec dictionary directory. Try another word!")
        else:
            # Store score into session history dictionary
            st.session_state.guess_history[user_input] = score_pct
            
            # Contextual thermal alerts based on your original exact conditions
            if score_pct > 65:
                st.error(f"🔥 **Extremely Hot!** '{user_input}' is very close in the Word2Vec space! ({score_pct}%)")
            elif score_pct > 35:
                st.warning(f"🌤️ **Getting Warmer!** '{user_input}' shares context. ({score_pct}%)")
            else:
                st.info(f"❄️ **Freezing Cold!** '{user_input}' is far off. ({score_pct}%)")

# Navigation Interface Control
col1, col2 = st.columns([4, 1])
with col1:
    if st.button("🔄 Next Word / Skip"):
        reset_game()
        st.rerun()
with col2:
    if st.button("🏳️ Give Up"):
        st.write(f"The secret word was: **{st.session_state.secret_word}**")
        st.session_state.game_over = True

# =====================================================================
# 📊 SIDEBAR HISTORY (Visual Feedback Queue Tracker)
# =====================================================================

if st.session_state.guess_history:
    st.sidebar.title("📊 Guess Proximity")
    st.sidebar.write("Sorted by Word2Vec vector distance:")
    
    # Render history queue sorted by score descending
    sorted_history = sorted(st.session_state.guess_history.items(), key=lambda x: x[1], reverse=True)
    
    for word, score in sorted_history:
        if score > 65:
            st.sidebar.markdown(f"🔴 **{word}** → `{score}%` (Hot)")
        elif score > 35:
            st.sidebar.markdown(f"🟡 **{word}** → `{score}%` (Warm)")
        else:
            st.sidebar.markdown(f"🔵 {word} → `{score}%` (Cold)")
