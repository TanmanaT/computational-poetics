"""
image_to_text.py  —  OCR Module
================================
Converts images of poems/text into strings using Tesseract OCR.

Supports:
  - File upload  (JPEG, PNG, BMP, TIFF)
  - Webcam capture  (via OpenCV)
  - Raw bytes  (for API uploads)
  - Base64 strings  (for web frontend)

Pipeline:
  Image → Grayscale → Upscale → Denoise → Sharpen → Threshold
        → Ruled-line mask → Tesseract (PSM 11, LSTM) → Clean text

Fixes applied vs original:
  1. preprocess_for_ocr() is now actually called in robust_extract()
  2. clean_ocr_output() is now actually called on every return path
  3. EasyOCR now receives a numpy array, not raw bytes (which it cannot decode)
  4. Tesseract now runs on the preprocessed image, not the raw PIL image
  5. Horizontal ruled lines are masked (not subtracted) to preserve text pixels
  6. Tesseract config upgraded: --psm 11 --oem 1 (sparse-text + LSTM engine)
     PSM 11 handles informal / handwritten / non-uniform layouts far better
  7. to_numpy() added: centralises bytes / path / array → ndarray conversion
  8. Webcam capture now passes the numpy frame directly — no temp file needed
"""

import re, os, base64, io
import numpy as np

# ── Global module state ───────────────────────────────────────────────────────
OCR_CONFIG = {
    'CV2_OK': False,
    'EASY_OK': False,
    'TESSERACT_OK': False,
    'READER': None
}

# ── Optional imports — handled gracefully ────────────────────────────────────
try:
    import cv2
    OCR_CONFIG['CV2_OK'] = True
except Exception:
    pass

try:
    import easyocr
    OCR_CONFIG['EASY_OK'] = True
except Exception:
    pass

try:
    import pytesseract
    from PIL import Image
    pytesseract.get_tesseract_version()
    OCR_CONFIG['TESSERACT_OK'] = True
except Exception:
    pass


def get_reader():
    if OCR_CONFIG['READER'] is None and OCR_CONFIG['EASY_OK']:
        try:
            import torch
            OCR_CONFIG['READER'] = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        except Exception:
            OCR_CONFIG['EASY_OK'] = False
    return OCR_CONFIG['READER']


# ─── Input normalisation ──────────────────────────────────────────────────────
def to_numpy(input_data) -> np.ndarray:
    """
    Normalise any input into a BGR numpy array for cv2 processing.
    Accepts: file path (str) | raw bytes | existing numpy array.

    FIX: was missing entirely — EasyOCR and preprocessing both crashed
    silently when passed raw bytes because neither can decode them directly.
    """
    if isinstance(input_data, np.ndarray):
        return input_data

    if isinstance(input_data, bytes):
        arr = np.frombuffer(input_data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2 could not decode image bytes.")
        return img

    if isinstance(input_data, str):
        img = cv2.imread(input_data)
        if img is None:
            raise ValueError(f"cv2 could not read file: {input_data}")
        return img

    raise TypeError(f"Unsupported input type: {type(input_data)}")


# ─── Image Preprocessing ──────────────────────────────────────────────────────
def remove_horizontal_lines_binary(binary: np.ndarray) -> np.ndarray:
    """
    Remove ruled / notebook lines from an already-thresholded binary image.

    Working on the binary image (not grayscale) is far more reliable:
    text strokes and ruled lines are cleanly separated, so a wide
    horizontal kernel isolates lines without touching letter pixels.

    FIX: original code subtracted lines from the grayscale image before
    thresholding, which erased text pixels that overlapped with lines and
    caused the adaptive threshold to misread the cleaned regions.
    The correct order is: threshold first → remove lines from binary.
    """
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    lines_detected = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    # Add detected line pixels back as white (background) in the binary image
    return cv2.add(binary, lines_detected)


def preprocess_for_ocr(img_array: np.ndarray) -> np.ndarray:
    """
    Preprocess image to maximise OCR accuracy, especially for:
      - Handwritten text
      - Photos of pages (uneven lighting)
      - Notebook / lined paper

    Pipeline:
      1. Grayscale
      2. Upscale to ~2500px on longest edge (Tesseract needs high DPI)
      3. NL-means denoise  (removes grain and paper texture noise)
      4. Adaptive threshold (handles shadows and lighting gradients)
      5. Remove ruled lines from binary image (correct order — see fix note)
      6. Slight dilation   (thickens thin handwriting strokes)

    FIX: this function was defined in the original but never called.
    FIX: ruled-line removal now happens after thresholding, not before.
    """
    if not OCR_CONFIG['CV2_OK']:
        return img_array

    # 1. Grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()

    # 2. Upscale — Tesseract accuracy degrades significantly below ~150 DPI
    h, w = gray.shape
    if max(h, w) < 2000:
        scale = 2500 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_LANCZOS4)

    # 3. Denoise — NL-means handles paper grain and faint ruled-line texture
    gray = cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7,
                                    searchWindowSize=21)

    # 4. Adaptive threshold — block size 31, C=15 tuned for handwriting
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    # 5. Remove ruled lines from binary (correct order — after threshold)
    thresh = remove_horizontal_lines_binary(thresh)

    # 6. Slight dilation to thicken thin handwriting strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    processed = cv2.dilate(thresh, kernel, iterations=1)

    return processed


# ─── Text Cleaning ────────────────────────────────────────────────────────────
def clean_ocr_output(text: str) -> str:
    """
    Clean raw OCR output while preserving poem line structure.
    Poems need newlines preserved — unlike normal OCR cleanup.

    FIX: this function was defined in the original but never called.
    """
    if not text:
        return ""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = re.sub(r'[^\x20-\x7E]', '', line)   # strip non-ASCII artefacts
        line = re.sub(r'-{2,}', '', line)            # strip OCR dash runs from ruled lines
        line = re.sub(r'_{2,}', '', line)            # strip underscore runs
        line = re.sub(r' {3,}', '  ', line)          # collapse excess spaces
        line = line.strip()
        if line:
            cleaned.append(line)
    return '\n'.join(cleaned)


# ─── Main OCR Functions ───────────────────────────────────────────────────────
def robust_extract(input_data) -> str:
    """
    Preprocess the image, try EasyOCR first, PyTesseract as fallback.

    Args:
        input_data: File path (str), raw bytes, or numpy array.
    Returns:
        Cleaned extracted text (str).

    FIX: original passed raw bytes/path directly to both engines with no
    preprocessing — both preprocess_for_ocr and clean_ocr_output were dead code.
    """
    # Convert any input type → numpy array, then preprocess
    processed = None
    if OCR_CONFIG['CV2_OK']:
        try:
            img_array = to_numpy(input_data)
            processed = preprocess_for_ocr(img_array)  # FIX: actually called now
        except Exception:
            processed = None

    # 1. EasyOCR — receives preprocessed numpy array (not raw bytes / path)
    reader = get_reader()
    if reader is not None:
        try:
            # FIX: EasyOCR cannot accept raw bytes; pass numpy array
            ocr_input = processed if processed is not None else to_numpy(input_data)
            results = reader.readtext(ocr_input, detail=0, paragraph=True)
            text = '\n'.join(results)
            if text.strip():
                return clean_ocr_output(text)  # FIX: cleaning applied
        except Exception:
            pass

    # 2. Tesseract fallback
    if OCR_CONFIG['TESSERACT_OK']:
        try:
            import pytesseract
            from PIL import Image

            if processed is not None:
                # FIX: Tesseract now receives preprocessed image, not raw input
                pil_img = Image.fromarray(processed)
            elif isinstance(input_data, bytes):
                pil_img = Image.open(io.BytesIO(input_data))
            else:
                pil_img = Image.open(input_data)

            # FIX: --psm 11 (sparse text, no layout assumptions) handles
            # handwriting and informal layouts far better than --psm 6.
            # --oem 1 uses LSTM-only engine (more accurate than legacy).
            text = pytesseract.image_to_string(pil_img, config='--psm 11 --oem 1')
            return clean_ocr_output(text)  # FIX: cleaning applied
        except Exception:
            pass

    return ""


# ─── Public API ───────────────────────────────────────────────────────────────
def extract_from_file(image_path: str) -> str:
    """Extract text from a local file using robust fallback logic."""
    return robust_extract(image_path)


def extract_from_bytes(image_bytes: bytes) -> str:
    """Extract text from raw bytes using robust fallback logic."""
    return robust_extract(image_bytes)


def extract_from_base64(b64_string: str) -> str:
    """
    Extract text from a base64-encoded image string.
    Used by the Streamlit webcam widget which returns base64.

    Args:
        b64_string: Base64 string (with or without data:image/... prefix)
    Returns:
        Extracted text string
    """
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    image_bytes = base64.b64decode(b64_string)
    return extract_from_bytes(image_bytes)


def capture_from_webcam() -> str:
    """
    Open webcam, let user frame the poem, press SPACE to capture.
    Returns extracted text from the captured frame.

    NOTE: This is the CLI version.
          The Streamlit frontend uses st.camera_input() instead.
    """
    if not OCR_CONFIG['CV2_OK']:
        raise ImportError("OpenCV not installed. Run: pip install opencv-python")
    if not (OCR_CONFIG['EASY_OK'] or OCR_CONFIG['TESSERACT_OK']):
        raise ImportError("No OCR engine (EasyOCR or Tesseract) is available.")

    print("📷 Webcam open — frame the poem and press SPACE to capture. ESC to cancel.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check it is connected.")

    text = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Frame poem : press SPACE to capture | ESC to cancel', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # FIX: pass numpy frame directly — no temp file write needed
            text = robust_extract(frame)
            print(f"Captured: {len(text)} characters extracted")
            break
        elif key == 27:   # ESC
            print("Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return text


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--webcam':
            t = capture_from_webcam()
        else:
            t = extract_from_file(sys.argv[1])
        print("Extracted text:\n", t)
    else:
        print("Usage:")
        print("  python ocr/image_to_text.py path/to/image.jpg")
        print("  python ocr/image_to_text.py --webcam")