# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2
# import json
# import tempfile
# import os
# import requests
# from safetensors.torch import load_file
# from gtts import gTTS
# from langdetect import detect
# from deep_translator import GoogleTranslator
# from pydantic import BaseModel

# # =========================
# # APP INIT
# # =========================
# app = FastAPI(title="Telugu OCR API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # =========================
# # CONFIG
# # =========================
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL_DIR = "models"
# MODEL_PATH = f"{MODEL_DIR}/model.safetensors"
# TOKENIZER_PATH = f"{MODEL_DIR}/tokenizer.json"

# # ✅ YOUR GOOGLE DRIVE LINKS
# MODEL_URL = "https://drive.google.com/uc?export=download&id=14Vpa6-DCmOA6m9mnjinD2OytA8-7iUzr"
# TOKENIZER_URL = "https://drive.google.com/uc?export=download&id=1qyGK1hW-UzdXBAPda-XkP9kSsnzFpgCI"

# # =========================
# # DOWNLOAD MODEL FILES
# # =========================
# def download_file(url, path):
#     if os.path.exists(path) and os.path.getsize(path) > 1000:
#         print(f"✅ {path} already exists")
#         return

#     print(f"⬇ Downloading {path}...")
#     r = requests.get(url)

#     # ❌ Prevent wrong HTML downloads
#     if "text/html" in r.headers.get("Content-Type", ""):
#         raise Exception("❌ Download failed — got HTML instead of file")

#     with open(path, "wb") as f:
#         f.write(r.content)

#     print("✅ Download complete")

# def setup_model_files():
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)

#     download_file(MODEL_URL, MODEL_PATH)
#     download_file(TOKENIZER_URL, TOKENIZER_PATH)

# # =========================
# # MODEL
# # =========================
# class CRNN(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
#             nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2),
#             nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(),
#             nn.Conv2d(256,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2,1)),
#             nn.Conv2d(256,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(),
#             nn.Conv2d(512,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2,1))
#         )
#         self.rnn = nn.LSTM(512*2, 256, num_layers=2, bidirectional=True)
#         self.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = self.cnn(x)
#         b, c, h, w = x.size()
#         x = x.view(b, c*h, w).permute(2, 0, 1)
#         x, _ = self.rnn(x)
#         return self.fc(x)

# ocr_model = None
# idx_to_char = None

# # =========================
# # LOAD MODEL ON STARTUP
# # =========================
# @app.on_event("startup")
# def load_model():
#     global ocr_model, idx_to_char

#     setup_model_files()

#     if os.path.getsize(TOKENIZER_PATH) == 0:
#         raise Exception("❌ tokenizer.json is empty")

#     vocab = json.load(open(TOKENIZER_PATH, encoding="utf-8"))
#     idx_to_char = {int(v): k for k, v in vocab.items()}

#     ocr_model = CRNN(len(vocab))
#     ocr_model.load_state_dict(load_file(MODEL_PATH))
#     ocr_model.to(DEVICE)
#     ocr_model.eval()

#     print("✅ Model Loaded Successfully")

# # =========================
# # OCR FUNCTIONS
# # =========================
# def preprocess(img):
#     img = cv2.resize(img, (128, 32))
#     img = img.astype(np.float32) / 255.0
#     img = np.expand_dims(img, 0)
#     img = np.expand_dims(img, 0)
#     return torch.tensor(img).to(DEVICE)

# def decode(pred):
#     prev = -1
#     result = []
#     for p in pred:
#         if p != prev and p != 0:
#             result.append(idx_to_char.get(p, ""))
#         prev = p
#     return "".join(result)

# def run_ocr(img):
#     with torch.no_grad():
#         out = ocr_model(preprocess(img))
#     pred = out.argmax(2).squeeze().cpu().numpy()
#     return decode(pred)

# # =========================
# # TRANSLATION
# # =========================
# def translate_text(text, target):
#     return GoogleTranslator(source='auto', target=target).translate(text)

# # =========================
# # REQUEST MODELS
# # =========================
# class TTSRequest(BaseModel):
#     text: str
#     target_language: str

# # =========================
# # API ENDPOINTS
# # =========================
# @app.get("/")
# def home():
#     return {"status": "running 🚀"}

# @app.post("/ocr-translate")
# async def ocr_translate(file: UploadFile = File(...), target_language: str = "en"):
#     contents = await file.read()
#     img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)

#     if img is None:
#         raise HTTPException(status_code=400, detail="Invalid image")

#     text = run_ocr(img)

#     detected_lang = "te"
#     try:
#         detected_lang = detect(text)
#     except:
#         pass

#     translated = translate_text(text, target_language)

#     return {
#         "ocr_text": text,
#         "translated": translated,
#         "detected_language": detected_lang
#     }

# @app.post("/tts")
# def tts(req: TTSRequest):
#     tts = gTTS(text=req.text, lang=req.target_language)
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#     tts.save(tmp.name)
#     return FileResponse(tmp.name, media_type="audio/mpeg")


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
import requests
from safetensors.torch import load_file
from gtts import gTTS
from deep_translator import GoogleTranslator
from pydantic import BaseModel

# =========================

# APP INIT

# =========================

app = FastAPI(title="Telugu OCR API")

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_methods=["*"],
allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/model.safetensors"
TOKENIZER_PATH = f"{MODEL_DIR}/tokenizer.json"

MODEL_URL = "https://drive.google.com/uc?export=download&id=14Vpa6-DCmOA6m9mnjinD2OytA8-7iUzr"
TOKENIZER_URL = "https://drive.google.com/uc?export=download&id=1qyGK1hW-UzdXBAPda-XkP9kSsnzFpgCI"

# =========================

# DOWNLOAD FILES (FIXED)

# =========================

def download_file(url, path):
if os.path.exists(path) and os.path.getsize(path) > 1000:
print(f"✅ {path} already exists")
return

```
print(f"⬇ Downloading {path}...")

session = requests.Session()
response = session.get(url, stream=True)

# Handle Google Drive large file
for key, value in response.cookies.items():
    if key.startswith("download_warning"):
        url = url + "&confirm=" + value
        response = session.get(url, stream=True)

if "text/html" in response.headers.get("Content-Type", ""):
    raise Exception("❌ Wrong file downloaded")

with open(path, "wb") as f:
    for chunk in response.iter_content(1024):
        if chunk:
            f.write(chunk)

print("✅ Download complete")
```

def setup_model_files():
os.makedirs(MODEL_DIR, exist_ok=True)
download_file(MODEL_URL, MODEL_PATH)
download_file(TOKENIZER_URL, TOKENIZER_PATH)

# =========================

# MODEL

# =========================

class CRNN(nn.Module):
def **init**(self, num_classes):
super().**init**()

```
    self.cnn = nn.Sequential(
        nn.Conv2d(1,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
        nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
        nn.Conv2d(128,256,3,1,1), nn.ReLU(),
        nn.Conv2d(256,256,3,1,1), nn.ReLU(), nn.MaxPool2d((2,1)),
        nn.Conv2d(256,512,3,1,1), nn.ReLU(),
        nn.Conv2d(512,512,3,1,1), nn.ReLU(), nn.MaxPool2d((2,1))
    )

    self.rnn = nn.LSTM(512*2, 256, num_layers=2, bidirectional=True)
    self.fc = nn.Linear(512, num_classes)

def forward(self, x):
    x = self.cnn(x)
    b, c, h, w = x.size()
    x = x.view(b, c*h, w).permute(2, 0, 1)
    x, _ = self.rnn(x)
    return self.fc(x)
```

ocr_model = None
idx_to_char = None

# =========================

# LOAD MODEL

# =========================

@app.on_event("startup")
def load_model():
global ocr_model, idx_to_char

```
setup_model_files()

if not os.path.exists(TOKENIZER_PATH):
    raise Exception("❌ tokenizer.json missing")

with open(TOKENIZER_PATH, encoding="utf-8") as f:
    vocab = json.load(f)

idx_to_char = {int(v): k for k, v in vocab.items()}

ocr_model = CRNN(len(vocab))
ocr_model.load_state_dict(load_file(MODEL_PATH))
ocr_model.to(DEVICE)
ocr_model.eval()

print("✅ Model Loaded Successfully")
```

# =========================

# OCR

# =========================

def preprocess(img):
img = cv2.resize(img, (128, 32))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, 0)
img = np.expand_dims(img, 0)
return torch.tensor(img).to(DEVICE)

def decode(pred):
prev = -1
result = []
for p in pred:
if p != prev and p != 0:
result.append(idx_to_char.get(p, ""))
prev = p
return "".join(result)

def run_ocr(img):
with torch.no_grad():
out = ocr_model(preprocess(img))
pred = out.argmax(2).squeeze().cpu().numpy()
return decode(pred)

# =========================

# API

# =========================

class TTSRequest(BaseModel):
text: str
target_language: str

@app.get("/")
def home():
return {"status": "running 🚀"}

@app.post("/ocr-translate")
async def ocr_translate(file: UploadFile = File(...), target_language: str = "en"):
contents = await file.read()

```
img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise HTTPException(400, "Invalid image")

text = run_ocr(img)

try:
    translated = GoogleTranslator(source='auto', target=target_language).translate(text)
except:
    translated = "Translation failed"

return {
    "ocr_text": text,
    "translated": translated,
    "detected_language": "te"
}
```

@app.post("/tts")
def tts(req: TTSRequest):
tts = gTTS(text=req.text, lang=req.target_language)
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
tts.save(tmp.name)
return FileResponse(tmp.name, media_type="audio/mpeg")
