# 🗣️ XTTS-Based Text-to-Speech (TTS) Flask Web App

This project is a production-ready Text-to-Speech (TTS) web app built with **Flask**, powered by **Coqui XTTS v2**, and fine-tuned using a custom voice sample. Users can input text and generate speech in a personalized voice.

---

## 🚀 Features

- 🔊 Fine-tuned voice synthesis with XTTS
- 🌐 Clean web UI with real-time generation
- 🌍 Multilingual support (e.g., English, Hindi)
- ⚙️ Flask backend + Render deployment ready

---

## 📁 Project Structure

```
.
├── app.py                    # Flask backend
├── templates/
│   └── index.html            # Web UI
├── model/                    # XTTS base model files (model.pth, config.json, vocab.json)
├── fine_tuned_model/         # Fine-tuned voice (xtts_finetuned.pth)
├── requirements.txt          # Python dependencies
├── .render.yaml              # Render deployment config
└── README.md
```

---

## 🛠️ Setup Locally

1. **Clone the repo**
```bash
git clone https://github.com/your-username/TTS-App.git
cd TTS-App
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Flask app**
```bash
python app.py
```

Then open your browser at `http://localhost:5050`

---

## 🚀 Deploy on Render

1. **Ensure you have a `.render.yaml`** at the root:

```yaml
services:
  - type: web
    name: tts-app
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PORT
        value: 10000
    plan: starter
```

2. **Push code to GitHub**

```bash
git init
git remote add origin https://github.com/your-username/TTS-App.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

3. **Deploy from Render:**

- Go to [https://render.com](https://render.com)
- Click **"New Web Service"**
- Connect your GitHub repo
- Hit deploy

---

## ⚠️ Notes

- Your fine-tuned latents must be placed in `fine_tuned_model/xtts_finetuned.pth`
- If deploying to **Render's free tier**, ensure:
  - Model files are small or downloaded at runtime
  - RAM usage remains below **512MB**, or upgrade to paid plan

---

## 📬 Contact

Built by [Your Name]  
📧 your.email@example.com  
⭐ Star the repo if you find it helpful!
