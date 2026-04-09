# 🚀 Telugu OCR Translator — Flutter + FastAPI

## Project Structure
```
ocr_app/
├── backend/
│   ├── main.py            ← FastAPI server (OCR + Translation + TTS)
│   ├── requirements.txt   ← Python dependencies
│   ├── render.yaml        ← Deploy to Render.com
│   └── models/
│       ├── model.safetensors   ← COPY YOUR MODEL HERE
│       └── tokenizer.json      ← COPY YOUR TOKENIZER HERE
│
└── flutter_app/
    ├── lib/main.dart      ← Full Flutter UI
    ├── pubspec.yaml       ← Flutter dependencies
    └── android/
        └── app/src/main/AndroidManifest.xml  ← Permissions
```

---

## PART 1 — Deploy the Backend (FastAPI on Render)

### Step 1 — Prepare backend folder
```bash
cd backend
mkdir models
# Copy your model files:
cp "path/to/model.safetensors" models/
cp "path/to/tokenizer.json"    models/
```

### Step 2 — Test locally first
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# Open: http://localhost:8000
# Test: http://localhost:8000/docs  (Swagger UI)
```

### Step 3 — Deploy to Render.com (FREE)
1. Go to https://render.com → Sign up free
2. Click **New → Web Service**
3. Connect your GitHub repo (push backend/ folder to GitHub first)
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free (or Starter for better speed)
5. Add a **Persistent Disk** (500MB - 5GB) mounted at `/opt/render/project/src/models`
6. Upload your model files to the disk via Render's dashboard
7. Click **Deploy**
8. Your API URL will be: `https://your-app-name.onrender.com`

> ⚡ **Alternative: Railway.app** — even easier, just drag-drop your folder

---

## PART 2 — Build the Flutter APK

### Step 1 — Install Flutter
```bash
# Download Flutter SDK
# Windows: https://flutter.dev/docs/get-started/install/windows
# Just extract and add to PATH

flutter doctor   # Check everything is ready
```

### Step 2 — Install Android Studio
- Download from: https://developer.android.com/studio
- Install Android SDK (API 33+)
- Accept licenses: `flutter doctor --android-licenses`

### Step 3 — Set your API URL
Open `flutter_app/lib/main.dart` line 10:
```dart
const String API_BASE = "https://YOUR-BACKEND.onrender.com";
// Change to your actual Render URL, e.g.:
const String API_BASE = "https://telugu-ocr-api.onrender.com";
```

### Step 4 — Get Flutter dependencies
```bash
cd flutter_app
flutter pub get
```

### Step 5 — Build APK
```bash
# Debug APK (for testing):
flutter build apk --debug

# Release APK (for real use):
flutter build apk --release

# APK location:
# build/app/outputs/flutter-apk/app-release.apk
```

### Step 6 — Install on your Android phone
```bash
# Enable Developer Options + USB Debugging on your phone
# Connect via USB cable, then:
flutter install

# OR manually copy the APK to your phone and open it
# (Enable "Install from unknown sources" in phone settings)
```

---

## PART 3 — How it Works

```
📱 Flutter App
     │
     │ POST /ocr-translate (image file)
     ▼
🌐 FastAPI Backend (Render.com)
     │
     ├── CRNN Model → Telugu OCR text
     ├── NLLB-600M → Translate to target language
     └── gTTS → Text to Speech MP3
     │
     ▼
📱 Flutter App displays results + plays audio
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/languages` | GET | List supported languages |
| `/ocr` | POST | Image → OCR text |
| `/translate` | POST | Text → Translated text |
| `/ocr-translate` | POST | Image → OCR + Translate (one step) |
| `/tts` | POST | Text → MP3 audio |

### Test the API (Swagger UI)
After deploying, visit: `https://your-app.onrender.com/docs`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `flutter: command not found` | Add Flutter bin/ to PATH |
| `Android SDK not found` | Install Android Studio + SDK |
| `Camera permission denied` | Go to phone Settings → Apps → Allow Camera |
| `API connection error` | Check your API_BASE URL in main.dart |
| `Model not found on server` | Upload model files to Render persistent disk |
| `Build APK fails` | Run `flutter doctor` and fix any issues shown |
| App installs but crashes | Check `flutter logs` for error details |

---

## Quick Test Without Backend

To test the Flutter app UI before your backend is ready,
change API_BASE to a mock server:

```dart
// Use this free test API to verify UI works:
const String API_BASE = "https://httpbin.org";
```

Then check the UI renders correctly, then swap in your real URL.
