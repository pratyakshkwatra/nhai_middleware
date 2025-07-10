# NHAI Middleware: Fusion Engine

> ⏱️ 30 MINUTES OF FOOTAGE → 10 SECONDS OF PROCESSING  
> Because waiting is overrated.

## What is this?

This is the core video - data synchronization and processing engine for our NHAI road inspection solution. It fuses sensor XLSX data and road survey video into meaningful insights - at speeds up to:

- 🚀 720× faster than legacy systems  
- ⚡️ 180× faster than real-time  

We tackled the biggest pain point first:  
🎥 Video and 📈 data didn’t speak to each other.  
Now they do - perfectly.

---

## ⚙️ Features

- 🔁 Ultra fast CSV - video frame fusion  
- 🧠 Smart frame skipping & adaptive probes (Fusion Engine)  
- 🔍 OCR extraction via Tesseract  
- 📈 Detection of roughness, rut, crack, and ravelling  
- 🧲 Handles paused or frozen frames with no delay  
- 🔌 Modular, extensible, and simple to run  

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/pratyakshkwatra/nhai_middleware.git
cd nhai-middleware
```

### 2. Create a virtual environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR
Tesseract is required for OCR functionality.

- Ubuntu / Debian:
```bash
sudo apt update
sudo apt install tesseract-ocr
```
- macOS (with Homebrew):
```bash
brew install tesseract
```
- Windows (Untested):
  - Download from Tesseract Wiki (UB Mannheim)
  - Ensure "Add to PATH" is checked during installation.

### 5. Install FFmpeg
FFmpeg is required for extracting frames from video.

- Ubuntu / Debian:
```bash
sudo apt install ffmpeg
```
- macOS:
```bash
brew install ffmpeg
```
- Windows:
  - Download from [FFmpeg.org](https://ffmpeg.org/download.html) and add it to your system PATH.

---

## 🛠️ Usage
```bash
python run.py
```

---

## 🗂️ Project Structure
```bash
nhai-middleware/
├── main.py
├── examples/
├── results/
├── requirements.txt
└── README.md
```

---

## 🚦 Performance

- Input: 30 min, 550MB (Compressed to ~ 30MB), 5fps video  
- Output: Processed in ~10 seconds  
- Benchmark: ~720× faster than legacy 2+ hr pipelines  
- Frame stalls? No problem. Adaptive skip + smart relink = instant progress

---

## 🧠 Fusion Engine: Intelligence Under the Hood

> Paused Video ≠ Paused Progress

- Detects video freezes and vehicle halts  
- Skips irrelevant frames automatically  
- Matches CSV data to frames without delay  
- Always 100% accurate - even during interruptions

---

## 🎥 Demo

### ![Demo Video](https://github.com/pratyakshkwatra/nhai_middleware/blob/main/assets/demo.mp4)
---

## 📄 License

MIT License

---

## 🤝 Contributing

We welcome contributions!  
Please open issues or submit pull requests.  
If you found this useful, consider giving a ⭐ on GitHub.

---

Built for India’s roads - and the people who inspect them.