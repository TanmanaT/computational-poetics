"""
predict.py  —  Inference Engine
================================
Loads all saved models and runs predictions on any input text.
Used by the API. Also usable standalone.
"""

import os, sys, re, pickle, nltk, json
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('universal_tagset', quiet=True)
except:
    pass
# Path hack: ensure src/core and src are accessible
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(FILE_DIR)
SRC_DIR = os.path.dirname(CORE_DIR)
for p in [CORE_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from preprocessing.build_features import stylometric, get_pos_tags, get_theme_lexicons

# Root Path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODELS = os.path.join(ROOT, 'saved_models')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── NEURAL BOILERPLATE (Imported from shared base) ───────────────────────────


# Load Transformer Model (cached for efficiency)
_ST_MODEL = None
def get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return _ST_MODEL


# ─── INFERENCE WRAPPERS ───────────────────────────────────────────────────────


def _load(f):
    p = os.path.join(MODELS, f)
    return pickle.load(open(p,'rb')) if os.path.exists(p) else None


def build_X(text, wv, cv):
    # Semantic Embeddings
    model = get_st_model()
    T = model.encode([text]).astype(np.float32)
    
    # Linguistic Features
    S = np.array(stylometric(text), dtype=np.float32).reshape(1,-1)
    P = np.array(get_pos_tags(text), dtype=np.float32).reshape(1,-1)
    M = np.hstack([T, S, P])
    
    # Textual Diversity (TF-IDF)
    W = wv.transform([text]).toarray().astype(np.float32)
    C = cv.transform([text]).toarray().astype(np.float32)
    return np.hstack([M, W, C])


class PoeticsPredictor:
    """
    Master Inference Engine for the Computational Poetics Analyzer.
    
    Handles the coordinated loading of 4 ensemble models and their 
    respective TF-IDF vectorizers. Encapsulates the entire pipeline 
    from raw text input to multi-label/regression output.
    """
    def __init__(self):
        self.genre_model  = _load('genre_model.pkl')
        self.genre_le     = _load('genre_le.pkl')
        self.genre_wv     = _load('genre_word_tfidf.pkl')
        self.genre_cv     = _load('genre_char_tfidf.pkl')

        self.fig_model    = _load('figurative_model.pkl')
        self.fig_le       = _load('figurative_le.pkl')
        self.fig_wv       = _load('figurative_word_tfidf.pkl')
        self.fig_cv       = _load('figurative_char_tfidf.pkl')

        self.emo_model    = _load('emotion_model.pkl')
        self.emo_le       = _load('emotion_le.pkl')
        self.emo_wv       = _load('emotion_word_tfidf.pkl')
        self.emo_cv       = _load('emotion_char_tfidf.pkl')

        self.qual_model   = _load('quality_model.pkl')
        self.qual_wv      = _load('quality_word_tfidf.pkl')
        self.qual_cv      = _load('quality_char_tfidf.pkl')

        n = sum(1 for m in [self.genre_model, self.fig_model,
                             self.emo_model, self.qual_model] if m)
        print(f"✅ {n}/4 models loaded")

    def _classify(self, text, model, le, wv, cv):
        if not all([model, le, wv, cv]):
            return None, {}
        X     = build_X(text, wv, cv)
        label = le.inverse_transform(model.predict(X))[0]
        proba = {}
        if hasattr(model, 'predict_proba'):
            # Clip outlier values and fill NaNs to prevent front-end crash
            try:
                p = model.predict_proba(X)[0]
                proba = {c: round(float(np.nan_to_num(v, nan=0.0)), 6) for c,v in zip(le.classes_, p)}
            except Exception as e:
                print(f"Warning: Predict proba failed: {e}")
                proba = {c: 0.0 for c in le.classes_}
        return label, proba

    def predict_genre(self, text):
        label, proba = self._classify(text, self.genre_model,
                                       self.genre_le, self.genre_wv, self.genre_cv)
        return {'genre': label, 'genre_probabilities': proba}

    def predict_figurative(self, text):
        label, proba = self._classify(text, self.fig_model,
                                       self.fig_le, self.fig_wv, self.fig_cv)
        return {'figurative_type': label, 'figurative_probabilities': proba}

    def predict_emotion(self, text):
        label, proba = self._classify(text, self.emo_model,
                                       self.emo_le, self.emo_wv, self.emo_cv)
        return {'emotion': label, 'emotion_probabilities': proba}

    def predict_quality(self, text):
        if not all([self.qual_model, self.qual_wv, self.qual_cv]):
            return {'quality_score': None, 'grade': None}
        X     = build_X(text, self.qual_wv, self.qual_cv)
        
        # Calculate the 6 extra dimensions required by the Quality Model
        # [grammar, vocab, unique_ratio, avg_len, avg_syl, strong_vocab]
        s = stylometric(text)
        wc = max(1, s[0])
        extra_features = np.array([
            8.0,                       # default grammar_score
            7.5,                       # default vocab_score
            s[2],                      # unique_word_ratio (TTR)
            s[3],                      # avg_word_length
            s[12],                     # avg_syllables_per_word
            s[14]                      # strong_vocab_ratio (lexical density)
        ], dtype=np.float32).reshape(1, -1)
        
        print(f"\n[Quality Metrics Extracted]")
        print(f"  Grammar Approximation : 8.0")
        print(f"  Vocab Approximation   : 7.5")
        print(f"  Unique Word Ratio     : {s[2]:.3f}")
        print(f"  Avg Word Length       : {s[3]:.2f}")
        print(f"  Avg Syllables/Word    : {s[12]:.2f}")
        print(f"  Strong Vocab Ratio    : {s[14]:.3f}")
        
        X_qual = np.hstack([X, extra_features])
        score = round(float(np.clip(self.qual_model.predict(X_qual)[0], 0, 10)), 2)
        grade = ('Excellent' if score>=8 else 'Good' if score>=6
                 else 'Average' if score>=4 else 'Poor')
        return {'quality_score': score, 'grade': grade}

    def analyze(self, text):
        """
        Runs a comprehensive audit across all poetic metrics.
        
        Args:
            text (str): Input poem.
            
        Returns:
            dict: Aggregated predictions for Genre, Emotion, Figuration, Quality, 
                  and detailed stylometric statistics.
        """
        result = {}
        result.update(self.predict_genre(text))
        result.update(self.predict_figurative(text))
        result.update(self.predict_emotion(text))
        result.update(self.predict_quality(text))
        s = stylometric(text)
        result['stylometrics'] = {
            'word_count':             int(s[0]),
            'vocabulary_richness':    round(s[2], 3),
            'avg_word_length':        round(s[3], 2),
            'line_count':             int(s[5]),
            'avg_line_length':        round(s[6], 2),
            'punctuation_density':    round(s[8], 4),
            'avg_syllables_per_word': round(s[12], 2),
            'strong_vocab_ratio':     round(s[14], 3),
        }
        return result
