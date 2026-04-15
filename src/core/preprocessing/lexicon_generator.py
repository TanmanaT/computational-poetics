import os
import json
import torch
import sys
import re
from transformers import pipeline

# Setup Paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LEX_PATH = os.path.join(ROOT, 'src', 'core', 'preprocessing', 'lexicons.json')

# Constants
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # Slightly larger for better accuracy, still small
TARGET_WORDS = 50

CATEGORIES = {
    "genres": ["Gothic", "Modern", "Romance", "Nature", "War", "Tragedy", "Horror"],
    "emotions": ["Joy", "Sadness", "Fear", "Love", "Peace", "Courage", "Surprise", "Anger"],
    "figurative": ["Simile", "Metaphor", "Sarcasm", "Idiom", "Personification", "Hyperbole", "Alliteration", "Irony", "Imagery"]
}

def log(msg):
    print(f"  [AI-LEX] {msg}")
    sys.stdout.flush()

def clean_word(w):
    # Remove quotes, weird punctuation at ends, and redundant whitespace
    w = re.sub(r'["\']', '', w)
    w = w.strip().lower()
    # Ensure it's not a sentence or a definition
    if len(w.split()) > 3: return None
    if "refers to" in w or "repetition" in w or "example:" in w: return None
    if not re.match(r'^[a-z\s-]+$', w): return None
    return w if len(w) > 1 else None

def generate_lexicons(overwrite=False):
    """
    Generates expanded thematic lexicons using a robust Qwen-1.5B model.
    """
    if os.path.exists(LEX_PATH) and not overwrite:
        log(f"Lexicons already exist at {LEX_PATH}. Skipping generation.")
        return

    log(f"Loading model: {MODEL_ID}...")
    try:
        # Load in 4-bit if possible to save RAM, but here we use float32/fp16 for quality
        device = 0 if torch.cuda.is_available() else -1
        generator = pipeline(
            "text-generation", 
            model=MODEL_ID, 
            device=device,
            torch_dtype=torch.float32 # safe for CPU
        )
        log("Model loaded successfully.")
    except Exception as e:
        log(f"CRITICAL ERROR: Failed to load LLM: {e}")
        return

    lexicons = {}

    total_items = sum(len(v) for v in CATEGORIES.values())
    processed_count = 0

    for cat_name, items in CATEGORIES.items():
        lexicons[cat_name] = {}
        for item in items:
            processed_count += 1
            log(f"[{processed_count}/{total_items}] Generating keywords for {item} ({cat_name[:-1]})...")
            
            # Explicit few-shot style prompt
            prompt = (
                f"<|im_start|>system\n"
                f"You are a poetic analysis expert. Provide a comma-separated list of exactly {TARGET_WORDS} distinct keywords or short phrases associated with the {cat_name[:-1]} '{item}'. "
                f"Include both archaic and contemporary poetic terms. Do not provide definitions or explanations. Format: word1, word2, word3...<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Give {TARGET_WORDS} keywords for {item}:<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            try:
                outputs = generator(
                    prompt, 
                    max_new_tokens=512, 
                    do_sample=True, 
                    temperature=0.7, 
                    top_p=0.9
                )
                
                text = outputs[0]['generated_text']
                if "<|im_start|>assistant\n" in text:
                    text = text.split("<|im_start|>assistant\n")[-1]
                
                # Split by commas and lines (sometimes models use bullet points)
                raw_words = re.split(r'[,\n\d\.]+', text)
                
                words = []
                for w in raw_words:
                    cleaned = clean_word(w)
                    if cleaned:
                        words.append(cleaned)
                
                # Deduplicate and sort
                final_words = sorted(list(set(words)))
                lexicons[cat_name][item.lower()] = final_words
                log(f"    Added {len(final_words)} validated keywords for {item}.")
                
                # Incremental Save
                with open(LEX_PATH, 'w') as f:
                    json.dump(lexicons, f, indent=4)
                
            except Exception as e:
                log(f"    ERROR generating {item}: {e}")
                lexicons[cat_name][item.lower()] = []

    log(f"Final lexicons saved to {LEX_PATH}")
    log("Lexicon expansion complete.")

if __name__ == "__main__":
    force = "--overwrite" in sys.argv
    generate_lexicons(overwrite=force)