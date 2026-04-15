# The Computational Poetics Engine: A Step-by-Step Deep Dive

This manual provides a comprehensive, structured guide to how the "Computational Poetics" system analyzes literature. It maps every phase of the pipeline to the actual files and code in your project.

---

## 🏗️ The 5-Part Journey
1. **The Spark**: Input & Vision
2. **The Foundation**: Cleaning & The Vocabulary Engine (**Qwen-2.5**)
3. **The Translation**: Turning Words into Numbers (**MiniLM**)
4. **The Three Brains**: Neural Architectures (**MLP, LSTM, CNN**)
5. **The Diagnosis**: Final Result & Visual Insight

---

## 🟢 Part 1: The Initial Spark (Input & Vision)
**Primary Files**: `src/frontend/app.py`, `src/core/ocr/image_to_text.py`

### Step 0.5: Direct Entry (The Dashboard) ⌨️
*   **Why we use this**: To allow anyone with a digital copy of a poem to start an analysis instantly.
*   **Example**: Copy-pasting a poem from a website like *PoetryFoundation.org*.
```python
# From src/frontend/app.py
text = st.text_area("Source Text:", placeholder="Paste a poem here...")
if st.button("🔍 Run Analysis"):
    result = call_analyze(text) # Triggers the pipeline
```

### Step 0: Vision (The OCR Engine) — [OPTIONAL] 👁️
*   **Why we use this**: To bridge the gap between physical paper (like a library book or a handwritten letter) and digital AI.
*   **Example**: Taking a photo of an old family poem and having the AI "type" it for you.
```python
# From src/core/ocr/image_to_text.py
# If neural glasses fail, we grab the magnifying glass!
def robust_extract(input_data):
    reader = get_reader() # Modern Neural Engine (EasyOCR)
    if reader:
        # It reads the "image" into text
    return pytesseract.image_to_string(input_data) # Legacy Fallback (Tesseract)
```

---

## 🟢 Part 2: The Foundation (Cleaning & Vocabulary)
**Primary Files**: `src/core/preprocessing/clean_data.py`, `lexicon_generator.py`

### The Digital Shower 🚿
*   **Why we use this**: To prevent the AI from being "distracted" by computer noise (like website tags) that aren't part of the actual poem.
*   **Example**: Turning `<div>Once upon a <b>midnight</b></div>` into `Once upon a midnight`.
```python
# From src/core/preprocessing/clean_data.py
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)      # Deletes HTML tags
    return text.strip()
```

### The Vocabulary Engine: Qwen-2.5 (The Autonomous Lexicon) 📜
*   **Why we use this**: Humans use thousands of different words to describe one theme. This LLM makes sure the computer knows that "ghastly," "tomb," and "shroud" are all linked to "Gothic."
*   **Example**: Qwen identifies 50 "War" words (like *trenches, bayonet, battlefield*) so our system knows when a poem is about combat.
```python
# Qwen prompt to generate 50 keywords for a category (e.g. "Gothic")
prompt = f"Provide exactly 50 distinct keywords associated with the category '{item}'"
```

---

## 🟢 Part 3: The Translation (Feature Engineering)
**Primary File**: `src/core/preprocessing/build_features.py`

### The Stylometric Blueprint 📐
*   **Why we use this**: The "shape" of a poem tells us as much as the words. Fast rhythm usually means action; slow rhythm means mourning.
*   **Example**: Short, 3-word lines usually signal an "Angry" or "Modernist" poem.
```python
# From src/core/preprocessing/build_features.py
def stylometric(text):
    # We turn the poem's 'shape' into numbers
    ttr = len(set(words)) / len(words) # Vocabulary variety
    return [wc, ttr, ...] # A mathematical "fingerprint"
```

### The "Meaning Map" (The Mini-LLM) 🌍
*   **Why we use this**: To understand the "Vibe" and "Subtext" which simple word counting misses.
*   **Example**: Seeing that *"The sky wept"* is a sad image even if the word "sad" is never used.
```python
# Sentence-Transformers (MiniLM) translate the 'mood' into a 384-digit code
embeddings = ST_Model.encode([poem_text])
```

---

## 🟢 Part 4: The Three Brains (The Neural Hub)
**Primary File**: `src/core/models/neural_base.py`

### 1. The MLP (The Generalist) 🧠
*   **Why we use it**: To find the simplest links.
*   **Example**: Short lines + Dark words + Sad emotion = Likely a Tragedy.

### 2. The Bi-LSTM (The Rhythmist) 🥁
*   **Why we use it**: It treats the data as a sequence, listening to the "beat" of the features over time.
*   **Example**: Detecting a pattern of long-short-long lines (iambic pentameter).

### 3. The 1D-CNN (The Pattern-Seeker) 🕵️
*   **Why we use it**: To identify localized "Aesthetic Thumbprints" that humans might miss.
*   **Example**: Recognizing a specific pattern of punctuation that uniquely defines "Modernist" verse.

---

## 🟢 Part 5: Diagnosis & Visual Insight
**Primary Files**: `src/core/models/predict.py`, `src/frontend/app.py`

### Visual Storytelling 📊
*   **Why we use this**: Because 3,400 raw numbers are impossible for humans to read. Graphs make the data "speak" to us.
*   **Example**: A **Radar Chart** that visually shows your poem has the same "lexical shape" as Shakespearian sonnets.
```python
# From src/frontend/app.py
# Turning numbers into beautiful interactive charts
st.plotly_chart(fig_sun) # Interactive "Poetic Flow"
```

---

## 🚀 Summary: The Flow
- **Input** (Paste/Photo) ➔ **Clean** (Shower) ➔ **Glossary** (Qwen) ➔ **Numbers** (Stylometrics) ➔ **Brains** (Vote) ➔ **Dashboard** (Chart).
