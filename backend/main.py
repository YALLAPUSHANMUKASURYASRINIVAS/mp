from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import tempfile
import os
import shutil
from safetensors.torch import load_file
from PIL import Image
from gtts import gTTS
from langdetect import detect
from deep_translator import GoogleTranslator
from pydantic import BaseModel
import io

app = FastAPI(title="Telugu OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_HEIGHT = 32
IMG_WIDTH = 128

MODEL_PATH = "models/model.safetensors"
TOKENIZER_PATH = "models/tokenizer.json"

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Punjabi": "pa"
}

# =========================
# CRNN MODEL
# =========================
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(256,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2,1))
        )
        self.rnn = nn.LSTM(512*2, 256, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.view(b, c*h, w).permute(2, 0, 1)
        x, _ = self.rnn(x)
        return self.fc(x)

# =========================
# LOAD MODEL ON STARTUP
# =========================
ocr_model = None
idx_to_char = None

@app.on_event("startup")
def load_model():
    global ocr_model, idx_to_char
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        vocab = json.load(open(TOKENIZER_PATH, encoding="utf-8"))
        idx_to_char = {int(v): k for k, v in vocab.items()}
        ocr_model = CRNN(len(vocab))
        ocr_model.load_state_dict(load_file(MODEL_PATH))
        ocr_model.to(DEVICE)
        ocr_model.eval()
        print("✅ OCR Model loaded")
    else:
        print("⚠️ Model files not found — using dummy OCR")

# =========================
# OCR HELPERS
# =========================
def preprocess(img_array):
    img = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)
    return torch.tensor(img).to(DEVICE)

def decode_prediction(pred):
    prev = -1
    output = []
    for p in pred:
        if p != prev and p != 0:
            output.append(idx_to_char.get(p, ""))
        prev = p
    return "".join(output)

def run_ocr(img_array):
    if ocr_model is None:
        return "Model not loaded — place model.safetensors and tokenizer.json in /models folder"
    with torch.no_grad():
        out = ocr_model(preprocess(img_array))
    pred = out.argmax(2).squeeze().cpu().numpy()
    return decode_prediction(pred)

# =========================
# TRANSLATION HELPER
# =========================
def translate_text(text: str, target_lang: str) -> str:
    # Try offline NLLB first, fallback to Google
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        nllb_map = {
            "en": "eng_Latn", "te": "tel_Telu", "hi": "hin_Deva",
            "ta": "tam_Taml", "kn": "kan_Knda", "ml": "mal_Mlym",
            "mr": "mar_Deva", "bn": "ben_Beng", "gu": "guj_Gujr", "pa": "pan_Guru"
        }
        tgt_code = nllb_map.get(target_lang, "eng_Latn")
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
            max_length=256
        )
        return tokenizer.decode(tokens[0], skip_special_tokens=True)
    except Exception:
        # Fallback to Google Translate (needs internet)
        return GoogleTranslator(source='auto', target=target_lang).translate(text)

# =========================
# API ENDPOINTS
# =========================

@app.get("/")
def root():
    return {"status": "Telugu OCR API is running 🚀"}

@app.get("/languages")
def get_languages():
    return {"languages": LANGUAGES}

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """Upload an image → returns extracted Telugu text"""
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        text = run_ocr(img)
        detected_lang = "te"
        try:
            detected_lang = detect(text)
        except Exception:
            pass

        return {
            "success": True,
            "ocr_text": text,
            "detected_language": detected_lang
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TranslateRequest(BaseModel):
    text: str
    target_language: str  # language code e.g. "en", "hi"

@app.post("/translate")
def translate_endpoint(req: TranslateRequest):
    """Translate text to target language"""
    try:
        translated = translate_text(req.text, req.target_language)
        return {
            "success": True,
            "original": req.text,
            "translated": translated,
            "target_language": req.target_language
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr-translate")
async def ocr_and_translate(
    file: UploadFile = File(...),
    target_language: str = "en"
):
    """Upload image → OCR → Translate in one step"""
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        ocr_text = run_ocr(img)
        detected_lang = "te"
        try:
            detected_lang = detect(ocr_text)
        except Exception:
            pass

        translated = translate_text(ocr_text, target_language)

        return {
            "success": True,
            "ocr_text": ocr_text,
            "detected_language": detected_lang,
            "translated": translated,
            "target_language": target_language
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
def tts_endpoint(req: TranslateRequest):
    """Convert text to speech → returns MP3 file"""
    try:
        tts = gTTS(text=req.text, lang=req.target_language)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        return FileResponse(
            tmp.name,
            media_type="audio/mpeg",
            filename="speech.mp3"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
