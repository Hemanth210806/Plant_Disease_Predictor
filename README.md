# 🌿 Plant Disease Predictor (AI)

A full-stack web application that uses Deep Learning (MobileNetV2) to identify plant diseases from leaf images and generates specific treatment remedies using the Groq API.

## 🚀 Features
- **AI Diagnosis**: High-accuracy classification for 13 distinct plant disease categories (Potato, Tomato).
- **Smart Remedies**: Dynamically generated treatment plans (Immediate, Organic, and Chemical) via Groq API.
- **Premium UI**: Modern, responsive interface with a mobile-first design.

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.9+
- A [Groq API Key](https://console.groq.com/keys)

### 2. Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Set your Groq API key:
- **Windows (CMD)**: `set GROQ_API_KEY=your_key_here`
- **Windows (PowerShell)**: `$env:GROQ_API_KEY="your_key_here"`
- **Linux/Mac**: `export GROQ_API_KEY=your_key_here`

### 4. Running Locally
```bash
python app.py
```
The app will be available at `http://127.0.0.1:5000`.

## 📦 Deployment
The project is ready for deployment on platforms like **Heroku**, **Render**, or **Vercel**.
- **Procfile**: Included for Gunicorn.
- **Requirements**: Optimized for production.
- **.gitignore**: Configured to exclude large data files and sensitive environment info.

---
Built with ❤️ using TensorFlow, Flask, and Groq.
