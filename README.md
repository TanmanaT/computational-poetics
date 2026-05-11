# Computational Poetics using NLP
**A High-Fidelity Neural Engine for Structural and Thematic Verse Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/Logic-Transformers-purple.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/UI-Glassmorphism-FF4B4B.svg)](https://streamlit.io/)

---

## Overview
**Computational Poetics using NLP** is a multi-modal high-fidelity system designed to deconstruct the structural, thematic, and emotional nuances of verse. By combining **Transformer-based semantic embeddings**, **Large Language Model (LLM) Lexicon Expansion**, **Stylometric linguistic features**, and **Computer Vision (OCR)**, it provides an unparalleled analytical "fingerprint" of any poem.

### Key Features
- **AI-Driven Lexicon Engine**: Dynamically expanding thematic keywords using local LLMs (Qwen-0.5B) for ultra-precise Genre and Emotion mapping.
- **Premium UI**: High-end glassmorphism design system with dynamic gradients and custom typography (`Outfit` & `Playfair Display`).
- **Neural Ensemble Inference**: Simultaneous analysis of **Genre**, **Figurative Language**, **Emotion**, and **Quality** using a stabilized ensemble of MLP, Bi-LSTM, and CNN heads (SVM-free).
- **Vision Engine**: Robust OCR extraction via **EasyOCR** and **Tesseract** fallback for handwritten or printed poems.
- **Stylometric Analysis**: Identifies 20 distinct linguistic markers, specifically measuring the frequency of literary devices like metaphors, similes, and alliterative patterns.

---

## Technical Architecture
The project has been restructured into a professional `src/` layout for maximum modularity and scalability.

```text
poetics_final/
├── data/                   # Raw & Processed datasets
├── saved_models/           # Serialized Neural & XGBoost models
├── src/
│   ├── core/               # Deep logic & orchestration
│   │   ├── models/         # Inference & Training engines
│   │   ├── ocr/            # Multi-engine Vision logic
│   │   └── preprocessing/  # Cleaning & Feature Engineering
│   ├── frontend/           # Premium Streamlit App
│   └── backend/            # high-performance FastAPI server
└── run.py                  # Unified Master Entry Point
```

---

## Installation & Setup

### 1. Python Environment
```bash
pip install -r requirements.txt
```

### 2. OCR Engines (Optional but Recommended)
- **Tesseract**: [Install Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and ensure it's in your system PATH.
- **EasyOCR**: Automatically installed via requirements. (Recommended for high-accuracy image processing).

### 3. NLTK Data
The system automatically downloads required lexicons, but you can pre-load them:
```python
import nltk
nltk.download(['wordnet', 'omw-1.4', 'universal_tagset', 'averaged_perceptron_tagger'])
```

---

## Unified Runner (CLI Usage)
The system is managed via a single `run.py` script.

| Command | Description |
| :--- | :--- |
| `python run.py --app` | Launch the **Streamlit Dashboard** (Default) |
| `python run.py --pipeline` | Execute the **End-to-End ML Pipeline** (Cleaning -> Features -> Training) |
| `python run.py --api` | Start the **FastAPI Analysis Backend** (localhost:8000) |
| `python run.py --pipeline --force` | Force a complete **Cold Rebuild** of all features and models |

---

## Neural Ensemble Performance

| Metric | Baseline | Neural Era (Current) | Gain |
| :--- | :--- | :--- | :--- |
| **Genre F1-Score** | 0.43 | **0.78** | **+81%** |
| **Figurative F1** | 0.81 | **0.88** | **+9%** |
| **Emotion F1** | 0.42 | **0.92** | **+119%** |
| **Quality R²** | 0.12 | **0.67** | **+5x** |

---

## Detailed Performance Specs
- **Embeddings**: `all-MiniLM-L6-v2` (384-D) for dense semantic mapping.
- **Stylometrics**: 20-feature vector incorporating phonological (alliteration) and structural (variance/TTR) signals.
- **Vision Fallback**: The OCR module intelligently switches between **EasyOCR (Neural Path)** and **Tesseract (Legacy Path)** based on environment availability and character confidence.

---

## License
This project is optimized for academic research and high-fidelity computational linguistics.