"""
clean_data.py  —  STEP 1
========================
Cleans all 4 datasets and saves to data/processed/

Run: python preprocessing/clean_data.py
"""

import re
import os
import sys
import pandas as pd

# Root path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW = os.path.join(ROOT, 'data', 'raw')
OUT = os.path.join(ROOT, 'data', 'processed')
os.makedirs(OUT, exist_ok=True)

def should_skip(src_file, out_file):
    """
    Determines if a data processing step can be skipped based on file modification times.
    
    Args:
        src_file (str): Path to the raw source file.
        out_file (str): Path to the expected processed output file.
        
    Returns:
        bool: True if the output file exists and is newer than the source file, False otherwise.
    """
    if '--force' in sys.argv:
        return False
    if not os.path.exists(out_file):
        return False
    # If source doesn't exist (unexpected), don't skip to allow error handling downstream
    if not os.path.exists(src_file):
        return False
    return os.path.getmtime(out_file) > os.path.getmtime(src_file)

VALID_GENRES = {'war', 'nature', 'romance', 'tragedy', 'gothic', 'horror', 'modern'}

EMOTION_MAP = {
    'anger':'anger',   'angry':'anger',
    'sad':'sadness',   'sadness':'sadness',
    'fear':'fear',     'scared':'fear',
    'joy':'joy',       'happy':'joy',       'happiness':'joy',
    'love':'love',
    'peace':'peace',   'peaceful':'peace',
    'courage':'courage','brave':'courage',
    'surprise':'surprise'
}


def clean_text(text):
    """
    Standardizes raw text by removing HTML tags, normalizing whitespace, and stripping 
    non-ASCII characters to ensure clean feature extraction downstream.
    
    Args:
        text (str): The raw text string.
        
    Returns:
        str: The sanitized text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove HTML tags (e.g., <br>, <p>)
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize carriage returns to standard newlines
    text = re.sub(r'\r\n|\r', '\n', text)
    # Strip non-ASCII characters to prevent embedding errors
    text = re.sub(r'[^\x0A\x20-\x7E]', '', text)
    # Collapse multiple spaces/tabs into a single space
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Collapse 3+ consecutive newlines into exactly two (paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def balance_df(df, target_col):
    """
    Performs Tiered Class Balancing to prevent majoritarian bias.
    
    1. Caps dominant classes (e.g. 'Nature') at a reasonable threshold.
    2. Oversamples minority classes (e.g. 'Gothic', 'War') to match the target.
    """
    if target_col not in df.columns: return df
    counts = df[target_col].value_counts()
    if len(counts) <= 1: return df
    
    # Tiered Balancing Logic: 
    # Aim for a target count that is the median of active classes, 
    # but not exceeding 1600 to keep the Nature class from drowning the others.
    target_count = min(int(counts.median() * 1.5), 1600)
    
    balanced_chunks = []
    for cls, count in counts.items():
        chunk = df[df[target_col] == cls]
        if count > target_count:
            # Undersample dominant class
            balanced_chunks.append(chunk.sample(n=target_count, random_state=42))
        else:
            # Oversample minority class
            balanced_chunks.append(chunk.sample(n=target_count, replace=True, random_state=42))
            
    return pd.concat(balanced_chunks).sample(frac=1.0, random_state=42).reset_index(drop=True)


def clean_genre():
    """
    Cleans and standardizes the primary Genre classification dataset.
    
    Filters out invalid genres, empty strings, and exceedingly short poems.
    Imputes missing values for quality scores and figurative flags to ensure
    consistent dataset dimensions.
    
    Returns:
        pd.DataFrame: The cleaned genre dataset.
    """
    print("\nCleaning GENRE dataset...")
    src = os.path.join(RAW, 'poetry_labeled_BALANCED_7k.csv')
    out = os.path.join(OUT, 'genre_clean.csv')
    
    if should_skip(src, out):
        print("\tAlready clean. Skipping.")
        return pd.read_csv(out)

    df = pd.read_csv(src)
    df = df.rename(columns={'Poem': 'text'})
    
    # Apply text standardization
    df['text'] = df['text'].apply(clean_text)
    df['genre'] = df['genre'].str.lower().str.strip()
    
    # Filter constraints: Valid genres, non-empty, unique, minimum length (8 words)
    df = df[df['genre'].isin(VALID_GENRES)]
    df = df[df['text'].str.len() > 0]
    df = df.drop_duplicates(subset=['text'])
    df = df[df['text'].str.split().str.len() >= 8]

    # Select and reorder columns of interest
    keep = ['text', 'genre', 'quality_score', 'figurative_primary',
            'word_count', 'line_count', 'tone']
    keep = [c for c in keep if c in df.columns]
    df = df[keep].reset_index(drop=True)
    
    # Impute missing continuous/categorical data
    if 'quality_score' in df.columns:
        df['quality_score'] = df['quality_score'].fillna(df['quality_score'].median())
    if 'figurative_primary' in df.columns:
        df['figurative_primary'] = df['figurative_primary'].fillna('none').str.lower().str.strip()

    df.to_csv(out, index=False)
    print(f"\tgenre_clean.csv : {df.shape} | Genres: {df['genre'].value_counts().to_dict()}")
    
    # Create a balanced version for training
    df_balanced = balance_df(df, 'genre')
    df_balanced.to_csv(os.path.join(OUT, 'genre_balanced.csv'), index=False)
    print(f"\tCreated genre_balanced.csv : {df_balanced.shape}")
    
    return df


def clean_figurative():
    """
    Cleans the Figurative Language dataset.
    
    Standardizes text, lowercases the target variable, and filters out extremely
    short texts (< 4 words) which lack sufficient context for figurative detection.
    
    Returns:
        pd.DataFrame: The cleaned figurative language dataset.
    """
    print("\nCleaning FIGURATIVE dataset...")
    src = os.path.join(RAW, 'figurative_language_dataset.csv')
    out = os.path.join(OUT, 'figurative_clean.csv')

    if should_skip(src, out):
        print("\tAlready clean. Skipping.")
        return pd.read_csv(out)

    df = pd.read_csv(src)
    df['text'] = df['text'].apply(clean_text)
    df['figurative_type'] = df['figurative_type'].str.lower().str.strip()
    
    # Remove duplicates and enforce minimum length requirement (safely handling NaNs)
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.len() > 0].drop_duplicates(subset=['text'])
    df = df[df['text'].apply(lambda x: len(str(x).split()) >= 4)]
    
    df = df[['text', 'figurative_type']].reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"\tfigurative_clean.csv : {df.shape} | Types: {df['figurative_type'].value_counts().to_dict()}")
    
    df_bal = balance_df(df, 'figurative_type')
    df_bal.to_csv(os.path.join(OUT, 'figurative_balanced.csv'), index=False)
    
    return df


def clean_emotion():
    """
    Cleans the Emotion classification dataset.
    
    Loads data from Excel, applies text cleaning, and maps raw emotion strings
    to a standardized set of target classes via the EMOTION_MAP dictionary to
    prevent overlapping/synonymous categories.
    
    Returns:
        pd.DataFrame: The cleaned emotion dataset.
    """
    print("\nCleaning EMOTION dataset...")
    src = os.path.join(RAW, 'Poem_with_Emotions__1_.xlsx')
    out = os.path.join(OUT, 'emotion_clean.csv')

    if should_skip(src, out):
        print("\tAlready clean. Skipping.")
        return pd.read_csv(out)

    df = pd.read_excel(src)
    df = df.rename(columns={'Poem': 'text', 'Emotion': 'emotion'})
    df['text']    = df['text'].apply(clean_text)
    
    # Map raw emotions to standardized categories (e.g. 'happy' -> 'joy')
    df['emotion'] = df['emotion'].str.lower().str.strip().map(EMOTION_MAP).fillna('unknown')
    
    # Filter valid rows (safely handling NaNs)
    df = df[df['emotion'] != 'unknown']
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.len() > 0].drop_duplicates(subset=['text'])
    df = df[df['text'].apply(lambda x: len(str(x).split()) >= 4)]
    
    df = df[['text', 'emotion']].reset_index(drop=True)
    df.to_csv(os.path.join(OUT, 'emotion_clean.csv'), index=False)
    
    # Create balanced version
    df_bal = balance_df(df, 'emotion')
    df_bal.to_csv(os.path.join(OUT, 'emotion_balanced.csv'), index=False)
    
    print(f"\temotion_clean.csv    : {df.shape} | Emotions: {df['emotion'].nunique()}")
    print(f"\tCreated emotion_balanced.csv : {df_bal.shape}")
    return df


def clean_quality():
    """
    Cleans the Quality Score regression dataset.
    
    Normalizes numeric rankings and parses associated textual content to prepare
    the dataset for continuous regression architectures.
    
    Returns:
        pd.DataFrame: The cleaned quality dataset.
    """
    print("\nCleaning QUALITY dataset...")
    src = os.path.join(RAW, 'Quality_score__1_.csv')
    out = os.path.join(OUT, 'quality_clean.csv')

    if should_skip(src, out):
        print("\tAlready clean. Skipping.")
        return pd.read_csv(out)

    df = pd.read_csv(src)
    # Prefer pre-corrected text column if available
    if 'corrected_text' in df.columns:
        df['text'] = df['corrected_text'].fillna(df['text'])
    else:
        df['text'] = df['text']
        
    df['text'] = df['text'].apply(clean_text)
    
    # Normalise ranking score from domain [28, 96.4] to range [0, 10]
    df['quality_score'] = (df['overall_score'] / 10.0).clip(0, 10)
    df['quality_label'] = df['quality_label'].str.lower().str.strip()
    
    # Filter valid rows (enforcing minimum poem length to ensure structural integrity)
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.len() > 0].drop_duplicates(subset=['text'])
    df = df[df['text'].apply(lambda x: len(str(x).split()) >= 10)]

    # Select and reorder columns of interest
    keep = ['text', 'quality_score', 'quality_label', 'grammar_score',
            'vocab_score', 'unique_word_ratio', 'avg_word_length',
            'avg_syllables_per_word', 'strong_vocab_ratio']
    keep = [c for c in keep if c in df.columns]
    
    df = df[keep].reset_index(drop=True)
    df.to_csv(os.path.join(OUT, 'quality_clean.csv'), index=False)
    print(f"\tquality_clean.csv : {df.shape} | Score: {df['quality_score'].min():.1f}–{df['quality_score'].max():.1f}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print("=" * 55)
    print("\tSTEP 1 - DATA CLEANING")
    print("=" * 55)
    
    # Override skip logic if --force is used
    if args.force:
        def should_skip(s, o): return False

    g = clean_genre()
    f = clean_figurative()
    e = clean_emotion()
    q = clean_quality()
    print("\nALL DONE")
    print(f"\tgenre_clean.csv      : {len(g):,} rows")
    print(f"\tfigurative_clean.csv : {len(f):,} rows")
    print(f"\temotion_clean.csv    : {len(e):,} rows")
    print(f"\tquality_clean.csv    : {len(q):,} rows")
