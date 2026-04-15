"""
build_features.py  —  STEP 2
==============================
Extracts and serializes dense numerical features from cleaned datasets.

OPTIMISED FOR 16GB RAM:
- Word TF-IDF: 3000 features
- Char TF-IDF: 2000 features
- Uses float32 throughout to prevent memory overflow
- Processes embeddings in chunks utilizing available GPU/CPU

Run: python preprocessing/build_features.py
"""

import os, sys, json
import re
import pickle
import nltk
from nltk.corpus import wordnet as wn
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('universal_tagset', quiet=True)
except:
    pass
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed
from tqdm import tqdm

# Root path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROC   = os.path.join(ROOT, 'data', 'processed')
MODELS = os.path.join(ROOT, 'saved_models')
os.makedirs(MODELS, exist_ok=True)

def should_skip(src_file, out_file):
    """
    Determines if a computationally expensive feature extraction step can be skipped.
    
    Args:
        src_file (str): Path to the cleaned source CSV file.
        out_file (str): Path to the target feature matrix (.npz file).
        
    Returns:
        bool: True if the extracted feature matrix already exists and is fresher than the source.
    """
    if '--force' in sys.argv:
        return False
    if not os.path.exists(out_file):
        return False
    if not os.path.exists(src_file):
        return False
    return os.path.getmtime(out_file) > os.path.getmtime(src_file)

# ─── NLP-Augmented Lexicons ──────────────────────────────────────────────────
_THEME_LEXICONS = None
def get_theme_lexicons():
    """Generates expanded semantic lexicons based on the target category labels."""
    global _THEME_LEXICONS
    if _THEME_LEXICONS is not None:
        return _THEME_LEXICONS
    
    # Path to LLM-generated Lexicons
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    LEX_PATH = os.path.join(ROOT, 'src', 'core', 'preprocessing', 'lexicons.json')
    
    lexicons = {}
    if os.path.exists(LEX_PATH):
        print(f"\t[Lexicon] Loading LLM-optimized library from {os.path.basename(LEX_PATH)}")
        with open(LEX_PATH, 'r') as f:
            data = json.load(f)
            # Flatten genres, emotions, and figurative into a single lookup dict
            for cat in data.values():
                for label, words in cat.items():
                    lexicons[label.lower()] = set(words)
    else:
        print("\t[Lexicon] JSON not found, falling back to static seeds...")
        from nltk.corpus import wordnet as wn
        genre_classes = ['nature', 'modern', 'romance', 'gothic', 'tragedy', 'war']
        emotion_classes = ['anger', 'fear', 'joy', 'love', 'peace', 'sadness', 'surprise', 'courage']
        all_categories = list(set(genre_classes + emotion_classes))
        
        overrides = {
            'gothic': ['gothic', 'castle', 'crypt', 'spectre', 'ruins', 'abbey', 'shroud', 'curse', 'monastery', 'mansion', 'abandoned', 'decaying', 'unnatural', 'ancestor', 'midnight', 'hollow', 'corridors', 'sin', 'phantom', 'tomb', 'sepulchre', 'vault', 'obsidian', 'cathedral', 'macabre', 'eerie', 'haunt', 'shadows', 'misty', 'gloom', 'eternal', 'dormant', 'labyrinth', 'gargoyle', 'velvet', 'ancient', 'whispers', 'raven', 'nocturne'],
            'modern': ['modern', 'neon', 'asphalt', 'digital', 'concrete', 'steel', 'internet', 'cyber', 'urban', 'machine', 'circuit', 'chrome', 'static', 'plastic', 'satellite', 'algorithm', 'virtual', 'skyscraper', 'factory', 'pollution', 'synthetic', 'pixel', 'microwave', 'radiation', 'metropolis', 'terminal', 'network'],
            'romance': ['romance', 'passion', 'beloved', 'embrace', 'ardent', 'devotion', 'sweetheart', 'longing', 'soulmate', 'eternal', 'tenderness', 'intimacy', 'heartbeat', 'destiny', 'starlight', 'velvet', 'roses', 'clandestine', 'whispers', 'adoration', 'enchanting', 'sublime', 'rapture', 'fidelity'],
            'nature': ['nature', 'forest', 'mountain', 'ocean', 'river', 'meadow', 'blossom', 'sunlight', 'breeze', 'valley', 'wildlife', 'scenery', 'horizon', 'flora', 'fauna', 'wilderness', 'landscape', 'terrain', 'ecology', 'verdant', 'azure', 'cascade', 'glade', 'willow', 'sapphire', 'zenith', 'twilight', 'amber', 'celestial', 'terra', 'organic', 'bloom', 'harvest', 'solstice', 'equate', 'rhythm', 'echo'],
            'war': ['war', 'battle', 'soldier', 'infantry', 'artillery', 'frontline', 'conflict', 'weapon', 'strategy', 'victory', 'defeat', 'casualty', 'veteran', 'tactical', 'blockade', 'invasion', 'trench', 'shell', 'grenade', 'sniper', 'bunker', 'bombardment', 'casualty', 'siege', 'ammunition', 'barracks', 'garrison', 'battalion', 'carnage', 'valor', 'martyr', 'attrition', 'scorched', 'liberation'],
            'tragedy': ['tragedy', 'mourn', 'grief', 'sorrow', 'lament', 'misery', 'misfortune', 'calamity', 'catastrophe', 'despair', 'agony', 'plight', 'woe', 'bleak', 'desolate', 'heartbreak', 'funeral', 'elegy', 'obituary', 'morbid', 'bereavement', 'anguish', 'grim', 'terminal', 'abyss', 'ruins', 'shattered', 'weeping', 'hollow', 'shadow', 'end', 'hollow', 'void']
        }
        
        for category in all_categories:
            seeds = overrides.get(category, [category])
            lexicon = set(seeds)
            for word in seeds:
                for syn in wn.synsets(word):
                    for lemma in syn.lemmas():
                        lexicon.add(lemma.name().lower().replace('_', ' '))
            lexicons[category] = lexicon
            
    _THEME_LEXICONS = lexicons
    return lexicons


# ─── Stylometic & Advanced Features ──────────────────────────────────────────
def get_pos_tags(text):
    """
    Calculates the normalized Part-of-Speech (POS) distribution of a given text.
    
    Relies on NLTK's `universal` tagset to map English words to 12 core syntactic
    categories (Verbs, Nouns, Adjectives, etc). Normalizes outputs to percentages
    summing to 1.0 to handle poems of varying lengths.
    
    Args:
        text (str): The raw poem string.
        
    Returns:
        list[float]: A 12-dimensional vector of POS frequency percentages.
    """
    tokens = nltk.word_tokenize(text.lower())
    if not tokens: 
        return [0.0] * 12
        
    tags = nltk.pos_tag(tokens, tagset='universal')
    tag_counts = nltk.FreqDist(tag for word, tag in tags)
    
    categories = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X', '.']
    return [tag_counts.get(cat, 0) / len(tokens) for cat in categories]


def stylometric(text):
    """
    Extracts core structural, rhythmic, and thematic numerical features from text.
    Standardized to 24-dimensions.
    
    Args:
        text (str): The raw poem string.
    Returns:
        list[float]: A 24-dimensional feature vector.
    """
    if not isinstance(text, str) or not text.strip():
        return [0.0] * 24
        
    # Extract purely alphabetic words for vocabulary metrics
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    wc = len(words)
    if wc == 0:
        return [0.0] * 24
    
    unique  = len(set(w.lower() for w in words))
    chars   = len(text)
    # Calculate structural layout
    lines   = [l.strip() for l in re.split(r'\n|[.!?]+', text) if l.strip()]
    lc      = max(1, len(lines))
    
    line_lengths = [len(l) for l in lines]
    structural_variance = np.std(line_lengths) / (np.mean(line_lengths) + 1e-6)
    comma_count = text.count(',')
    semicolon_count = text.count(';')
    dash_count = text.count('-') + text.count('—')
    
    def syl(w):
        c = len(re.findall(r'[aeiouy]+', w.lower()))
        if w.endswith('e') and c > 1: c -= 1
        return max(1, c)

    syllables = [syl(w) for w in words]
    
    # Richness / Complexity
    ttr = unique / wc
    lexical_density = sum(1 for w in words if len(w) > 5) / wc
    
    # Dynamic Theme Markers (Dedicated Genre Anchors)
    lexicons = get_theme_lexicons() or {}
    
    def get_score(key):
        lex = lexicons.get(key, set())
        if not lex: return 0.0
        return sum(1 for w in words if w.lower() in lex) / wc

    modern_score  = get_score('modern')
    war_score     = get_score('war')
    nature_score  = get_score('nature')
    gothic_score  = get_score('gothic')
    romance_score = get_score('romance')
    tragedy_score = get_score('tragedy')
    
    # Figurative Structural Markers
    allit_count = 0
    consonants = "bcdfghjklmnpqrstvwxyz"
    for i in range(len(words)-1):
        w1, w2 = words[i].lower(), words[i+1].lower()
        if len(w1) > 2 and len(w2) > 2 and w1[0] == w2[0] and w1[0] in consonants:
            allit_count += 4 # High precision weight
            
    # 2. Simile: Presence of comparison particles
    low_text = text.lower()
    simile_markers = [" like ", " as ", " than "]
    simile_count = sum(low_text.count(m) for m in simile_markers)
    
    # 3. Metaphor/Personification: Copula verbs or animate actions on inanimate nouns
    metaphor_markers = [" is a ", " was a ", " are a ", " were a "]
    metaphor_count = sum(low_text.count(m) for m in metaphor_markers)

    allit_score = min(1.0, allit_count / wc)
    simile_score = min(1.0, (simile_count * 5) / wc) # Multiplied to give signal weight
    metaphor_score = min(1.0, (metaphor_count * 5) / wc)

    return [
        float(wc),                                          # 0
        float(unique),                                      # 1
        ttr,                                                # 2
        sum(len(w) for w in words) / wc,                    # 3
        float(chars),                                       # 4
        float(lc),                                          # 5
        wc / lc,                                            # 6
        structural_variance,                                # 7
        sum(1 for c in text if c in '.,!?;:') / chars,      # 8
        comma_count / chars,                                # 9
        semicolon_count / chars,                            # 10
        dash_count / chars,                                 # 11
        sum(syllables) / wc,                                # 12 (Avg Syl)
        max(syllables) if syllables else 0,                  # 13 (Max Syl)
        lexical_density,                                    # 14
        modern_score,                                       # 15
        war_score,                                          # 16
        nature_score,                                       # 17
        gothic_score,                                       # 18
        romance_score,                                      # 19
        tragedy_score,                                      # 20
        allit_score,                                        # 21
        simile_score,                                       # 22
        metaphor_score                                      # 23
    ]


# Global caches to avoid redundant computing
CACHE_FILE = os.path.join(PROC, 'embedding_cache.pkl')
LING_CACHE_FILE = os.path.join(PROC, 'linguistic_cache.pkl')
_EMBED_CACHE = {}
_LING_CACHE = {}
_ST_MODEL = None

def load_caches():
    """
    Loads pre-computed semantic embeddings and linguistic features from disk into memory.
    
    Prevents redundant computation of expensive operations (like SentenceTransformer 
    encoding or NLP tagging) across multiple pipeline runs.
    """
    global _EMBED_CACHE, _LING_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                _EMBED_CACHE = pickle.load(f)
            print(f"\tLoaded {len(_EMBED_CACHE):,} embeddings from cache.")
        except Exception as e: print(f"\tError loading embedding cache: {e}")
    
    if os.path.exists(LING_CACHE_FILE):
        try:
            with open(LING_CACHE_FILE, 'rb') as f:
                _LING_CACHE = pickle.load(f)
            print(f"\tLoaded {len(_LING_CACHE):,} linguistic features from cache.")
        except Exception as e: print(f"\tError loading linguistic cache: {e}")

def save_caches():
    """
    Serializes the in-memory coordinate dictionaries back to disk.
    
    Called after processing new texts to append newly computed vectors to 
    the persistent cache structures.
    """
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(_EMBED_CACHE, f)
        with open(LING_CACHE_FILE, 'wb') as f:
            pickle.dump(_LING_CACHE, f)
        print(f"\tCaches synchronized.")
    except Exception as e:
        print(f"\tError saving caches: {e}")

def get_st_model():
    """
    Lazily loads the SentenceTransformer model into memory.
    
    Uses standard MiniLM architecture (384 dimensions). Binds execution to a CUDA GPU 
    if available, otherwise falls back to multicore CPU routing.
    
    Returns:
        SentenceTransformer: The compiled PyTorch execution model.
    """
    global _ST_MODEL
    if _ST_MODEL is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\tLoading Sentence-Transformer (all-MiniLM-L6-v2) on {device}...")
        _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return _ST_MODEL

def extract_linguistic(t):
    """
    Adapter function combining stylometric and POS extraction for multiprocessing.
    
    Args:
        t (str): A single poem text.
    Returns:
        tuple(list, list): (Stylometric feature vector, POS feature vector)
    """
    return stylometric(t), get_pos_tags(t)

def build_matrix(texts):
    """
    Constructs the Dense Feature Matrix (DFM) mapping input texts to numerical rows.
    
    Orchestrates the feature pipeline: 
    1) Semantic Sentence Transformer embeddings (Cached)
    2) Stylometric structure extractions (Cached/Parallelized)
    3) Part-of-Speech syntax distributions (Cached/Parallelized)
    
    Args:
        texts (pd.Series): The raw textual input series.
        
    Returns:
        np.ndarray: A horizontally stacked float32 matrix incorporating all 3 feature spaces.
    """
    print(f"\tComputing semantic embeddings + stylometrics + POS for {len(texts):,} texts...")
    
    # 1. Transformers (Semantic Sentence Embeddings)
    if not _EMBED_CACHE or not _LING_CACHE:
        load_caches()
    
    texts_list = texts.tolist()
    model = get_st_model()
    
    # Identify which texts need encoding
    to_encode = []
    indices_to_encode = []
    embedding_results = [None] * len(texts_list)
    
    for idx, txt in enumerate(texts_list):
        if txt in _EMBED_CACHE:
            embedding_results[idx] = _EMBED_CACHE[txt]
        else:
            to_encode.append(txt)
            indices_to_encode.append(idx)
            
    if to_encode:
        print(f"\tEncoding {len(to_encode):,} new semantic vectors...")
        new_vecs = model.encode(to_encode, show_progress_bar=True, batch_size=32).astype(np.float32)
        for i, vec in zip(indices_to_encode, new_vecs):
            _EMBED_CACHE[texts_list[i]] = vec
            embedding_results[i] = vec
    else:
        print("\tUsing cached semantic embeddings.")
    
    T = np.array(embedding_results).astype(np.float32)
    
    # 2. Stylometrics + POS
    to_ling = []
    ling_indices = []
    ling_results = [None] * len(texts_list)

    for idx, txt in enumerate(texts_list):
        cached = _LING_CACHE.get(txt)
        # Check if the cached feature vector has the correct new dimensionality (24)
        if cached and len(cached[0]) == 24:
            ling_results[idx] = cached
        else:
            to_ling.append(txt)
            ling_indices.append(idx)

    if to_ling:
        print(f"\tExtracting {len(to_ling):,} new linguistic features...")
        # Throttle n_jobs to 4 to save RAM on 16GB machines
        n_workers = min(4, os.cpu_count())
        new_ling = Parallel(n_jobs=n_workers, batch_size=50)(
            delayed(extract_linguistic)(t) for t in to_ling
        )
        for i, res in zip(ling_indices, new_ling):
            _LING_CACHE[texts_list[i]] = res
            ling_results[i] = res
        save_caches()
    else:
        print("\tUsing cached linguistic features.")

    S = [r[0] for r in ling_results]
    P = [r[1] for r in ling_results]
    
    M = np.hstack([T, np.array(S), np.array(P)]).astype(np.float32)
    return np.nan_to_num(M)


def build_tfidf(texts, wf=3000, cf=2000, name=''):
    """
    Extracts high-dimensional lexical features utilizing both Word and Character n-grams.
    
    Employs sublinear TF scaling to dampen the effect of highly frequent tokens.
    Feature counts are intentionally capped (wf=3000, cf=2000) to prevent Out-Of-Memory
    (OOM) crashes on 16GB RAM machines when training models downstream.
    
    Args:
        texts (pd.Series): The text corpus.
        wf (int): Maximum Word TF-IDF features to retain.
        cf (int): Maximum Character TF-IDF features to retain.
        name (str): Prefix used to save the fitted vectorizers to disk.
        
    Returns:
        tuple: (Word TF-IDF array, Character TF-IDF array) as float32 matrices.
    """
    print(f"\tWord TF-IDF ({wf} features)...")
    wv = TfidfVectorizer(max_features=wf, ngram_range=(1,2),
                         sublinear_tf=True, min_df=2,
                         analyzer='word',
                         token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b')
    W = wv.fit_transform(texts.fillna('')).toarray().astype(np.float32)

    print(f"\tChar TF-IDF ({cf} features)...")
    cv = TfidfVectorizer(max_features=cf, ngram_range=(3,5),
                         sublinear_tf=True, min_df=3, analyzer='char_wb')
    C = cv.fit_transform(texts.fillna('')).toarray().astype(np.float32)

    if name:
        pickle.dump(wv, open(os.path.join(MODELS, f'{name}_word_tfidf.pkl'), 'wb'))
        pickle.dump(cv, open(os.path.join(MODELS, f'{name}_char_tfidf.pkl'), 'wb'))
        print(f"\tSaved {name} vectorizers")
    return W, C


def combine(S, W, C):
    """Horizontally stacks Structural, Word, and Character matrices into a single array."""
    res = np.hstack([S, W, C]).astype(np.float32)
    return np.nan_to_num(res)


# ── Genre ─────────────────────────────────────────────────────────────────────
def genre_features():
    """
    Constructs the feature space for the Genre classification task.
    Builds stylometric embeddings, word TF-IDF, and char TF-IDF, concatenates
    them, and saves the resulting matrices/encoders to the processed data directory.
    """
    df_src = os.path.join(PROC, 'genre_balanced.csv')
    if not os.path.exists(df_src): df_src = os.path.join(PROC, 'genre_clean.csv')
    df = pd.read_csv(df_src)
    S  = build_matrix(df['text'])
    W, C = build_tfidf(df['text'], wf=2000, cf=1000, name='genre')
    X  = combine(S, W, C)
    le = LabelEncoder()
    y  = le.fit_transform(df['genre'])
    pickle.dump(le, open(os.path.join(MODELS, 'genre_le.pkl'), 'wb'))
    np.savez_compressed(os.path.join(PROC, 'genre_X.npz'), X=X)
    np.save(os.path.join(PROC, 'genre_y.npy'), y)
    print(f"\t{X.shape} Classes: {list(le.classes_)}")


# ── Figurative ────────────────────────────────────────────────────────────────
def figurative_features():
    """
    Constructs the feature space for the Figurative classification task.
    Saves the final DFM (Dense Feature Matrix) as `figurative_X.npz`.
    """
    print("\nFIGURATIVE features...")
    df_src = os.path.join(PROC, 'figurative_balanced.csv')
    if not os.path.exists(df_src): df_src = os.path.join(PROC, 'figurative_clean.csv')
    df = pd.read_csv(df_src)
    S  = build_matrix(df['text'])
    W, C = build_tfidf(df['text'], wf=2000, cf=1000, name='figurative')
    X  = combine(S, W, C)
    le = LabelEncoder()
    y  = le.fit_transform(df['figurative_type'])
    pickle.dump(le, open(os.path.join(MODELS, 'figurative_le.pkl'), 'wb'))
    np.savez_compressed(os.path.join(PROC, 'figurative_X.npz'), X=X)
    np.save(os.path.join(PROC, 'figurative_y.npy'), y)
    print(f"\t{X.shape} Classes: {list(le.classes_)}")


# ── Emotion ───────────────────────────────────────────────────────────────────
def emotion_features():
    """
    Constructs the feature space for the Emotion classification task.
    Note: Uses slightly reduced feature caps (2000 wf, 1500 cf) due to smaller dataset size.
    """
    print("\nEMOTION features...")
    df_src = os.path.join(PROC, 'emotion_balanced.csv')
    if not os.path.exists(df_src): df_src = os.path.join(PROC, 'emotion_clean.csv')
    df = pd.read_csv(df_src)
    S  = build_matrix(df['text'])
    W, C = build_tfidf(df['text'], wf=2000, cf=1000, name='emotion')
    X  = combine(S, W, C)
    le = LabelEncoder()
    y  = le.fit_transform(df['emotion'])
    pickle.dump(le, open(os.path.join(MODELS, 'emotion_le.pkl'), 'wb'))
    np.savez_compressed(os.path.join(PROC, 'emotion_X.npz'), X=X)
    np.save(os.path.join(PROC, 'emotion_y.npy'), y)
    print(f"\t{X.shape} Classes: {list(le.classes_)}")


# ── Quality ───────────────────────────────────────────────────────────────────
def quality_features():
    """
    Constructs the feature space for the Quality Score regression task.
    Extracts numerical embeddings mapping to the continuous [0, 10] quality metric.
    """
    print("\nQUALITY features...")
    df = pd.read_csv(os.path.join(PROC, 'quality_clean.csv'))
    S  = build_matrix(df['text'])
    W, C = build_tfidf(df['text'], wf=2000, cf=1000, name='quality')
    X  = combine(S, W, C)
    # Append extra numeric cols already in dataset
    extra_cols = ['grammar_score','vocab_score','unique_word_ratio',
                  'avg_word_length','avg_syllables_per_word','strong_vocab_ratio']
    extra_cols = [c for c in extra_cols if c in df.columns]
    if extra_cols:
        E = df[extra_cols].fillna(0).values.astype(np.float32)
        X = np.hstack([X, E])
    y = df['quality_score'].values.astype(np.float32)
    np.savez_compressed(os.path.join(PROC, 'quality_X.npz'), X=X)
    np.save(os.path.join(PROC, 'quality_y.npy'), y)
    print(f"\t{X.shape} Score range: {y.min():.1f}–{y.max():.1f}")


if __name__ == "__main__":
    print("=" * 55)
    print("\tSTEP 2 - FEATURE ENGINEERING")
    print("=" * 55)
    
    # ── Genre ─────────────────────────────────────────────────────────────────
    print("\nGENRE features...")
    df_path = os.path.join(PROC, 'genre_balanced.csv')
    if not os.path.exists(df_path): df_path = os.path.join(PROC, 'genre_clean.csv')
    out_path = os.path.join(PROC, 'genre_X.npz')
    
    if should_skip(df_path, out_path):
        print("\tMatrices already up-to-date. Skipping.")
    else:
        genre_features()

    # ── Figurative ────────────────────────────────────────────────────────────
    df_path = os.path.join(PROC, 'figurative_balanced.csv')
    if not os.path.exists(df_path): df_path = os.path.join(PROC, 'figurative_clean.csv')
    out_path = os.path.join(PROC, 'figurative_X.npz')

    if should_skip(df_path, out_path):
        print("\tMatrices already up-to-date. Skipping.")
    else:
        figurative_features()

    # ── Emotion ───────────────────────────────────────────────────────────────
    df_path = os.path.join(PROC, 'emotion_clean.csv')
    out_path = os.path.join(PROC, 'emotion_X.npz')

    if should_skip(df_path, out_path):
        print("\tMatrices already up-to-date. Skipping.")
    else:
        emotion_features()

    # ── Quality ───────────────────────────────────────────────────────────────
    df_path = os.path.join(PROC, 'quality_clean.csv')
    out_path = os.path.join(PROC, 'quality_X.npz')

    if should_skip(df_path, out_path):
        print("\tMatrices already up-to-date. Skipping.")
    else:
        quality_features()

    print("\nFEATURES SAVED to data/processed/")
