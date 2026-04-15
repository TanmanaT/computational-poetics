"""
api.py  —  FastAPI Backend
===========================
All endpoints:
  GET  /health
  POST /analyze          full analysis
  POST /genre
  POST /figurative
  POST /emotion
  POST /quality
  POST /upload_text      .txt file
  POST /upload_image     image file → OCR → analyze
  GET  /model_results

Run: uvicorn backend.api:app --reload --port 8000
"""

import sys, os, json
# Root path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'core'))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.predict import PoeticsPredictor

app = FastAPI(
    title="Computational Poetics Analyzer",
    description="Genre · Figurative Language · Emotion · Quality  +  OCR",
    version="3.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

predictor = PoeticsPredictor()
# Root is two levels up from src/backend
MODELS_DIR = os.path.join(ROOT, 'saved_models')


class TextReq(BaseModel):
    text: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "genre":      predictor.genre_model is not None,
            "figurative": predictor.fig_model   is not None,
            "emotion":    predictor.emo_model   is not None,
            "quality":    predictor.qual_model  is not None,
        }
    }


@app.post("/analyze")
def analyze(req: TextReq):
    if not req.text.strip():
        raise HTTPException(400, "Text is empty.")
    try:
        return predictor.analyze(req.text)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/genre")
def genre(req: TextReq):
    return predictor.predict_genre(req.text)


@app.post("/figurative")
def figurative(req: TextReq):
    return predictor.predict_figurative(req.text)


@app.post("/emotion")
def emotion(req: TextReq):
    return predictor.predict_emotion(req.text)


@app.post("/quality")
def quality(req: TextReq):
    return predictor.predict_quality(req.text)


@app.post("/upload_text")
async def upload_text(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        raise HTTPException(400, "Only .txt files.")
    content = await file.read()
    text = content.decode('utf-8', errors='replace')
    return predictor.analyze(text)


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Accept an image, run OCR, then full analysis."""
    try:
        from ocr.image_to_text import extract_from_bytes
    except ImportError:
        raise HTTPException(501, "OCR unavailable. Install: pip install pytesseract Pillow opencv-python")

    content = await file.read()
    try:
        text = extract_from_bytes(content)
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {e}")

    if not text.strip():
        raise HTTPException(422, "No text extracted from image.")

    result = predictor.analyze(text)
    result['ocr_extracted_text'] = text
    return result


@app.get("/model_results")
def model_results():
    p = os.path.join(MODELS_DIR, 'training_results.json')
    if os.path.exists(p):
        return json.load(open(p))
    return {"message": "Run python models/train_models.py first."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
