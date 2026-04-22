import os
import sqlite3
import datetime
import webbrowser
import subprocess
import sys
from threading import Timer
import json

# --- AUTO-DEPENDENCY INSTALLER ---
def install_dependencies():
    required = ['flask', 'google-generativeai', 'pillow', 'numpy', 'werkzeug', 'python-dotenv']
    for lib in required:
        try:
            __import__(lib.replace('-', '_'))
        except ImportError:
            print(f"Installing missing dependency: {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

install_dependencies()

# Now imports can proceed
from flask import Flask, request, render_template_string, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import google.generativeai as genai

# --- CONFIGURATION ---
# Baked-in API key for "Just Works" portability
GEMINI_API_KEY = "AIzaSyBoFzArTRMa9WDHKj7Mp6GKuGzpxU4Zpfs"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

UPLOAD_FOLDER = 'uploads'
DATABASE = 'waste_management.db'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  prediction TEXT,
                  raw_label TEXT,
                  confidence REAL,
                  timestamp TEXT,
                  user_feedback TEXT)''')
    conn.commit()
    conn.close()

init_db()

def predict_image(img_path):
    try:
        img = Image.open(img_path)
        prompt = """
        Analyze this image of waste. Categorize it into EXACTLY one:
        1. Recyclable Waste
        2. Wet Waste
        3. Hazardous Waste
        4. Dry Waste
        Provide a one-sentence explanation.
        Return ONLY a JSON object: {"category": "Name", "detected_item": "Item", "confidence": 0.95, "explanation": "..."}
        """
        response = model.generate_content([prompt, img])
        text_response = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(text_response)
        return data.get("category"), float(data.get("confidence")), data.get("detected_item"), data.get("explanation")
    except Exception as e:
        return "Dry Waste", 0.5, "Error", str(e)

# --- UI TEMPLATE ---
BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EcoSort AI | Portable Edition</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #10b981; --primary-dark: #059669; --bg: #0f172a; --card-bg: rgba(30, 41, 59, 0.7); --text: #f8fafc; --text-muted: #94a3b8; --accent: #38bdf8; --danger: #ef4444; --warning: #f59e0b; }
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Outfit', sans-serif; }
        body { background-color: var(--bg); background-image: radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.05) 0%, transparent 40%), radial-gradient(circle at 90% 80%, rgba(56, 189, 248, 0.05) 0%, transparent 40%); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
        nav { width: 100%; padding: 2rem; display: flex; justify-content: space-between; align-items: center; max-width: 1200px; }
        .logo { font-size: 1.5rem; font-weight: 800; display: flex; align-items: center; gap: 10px; }
        .logo span { color: var(--primary); }
        .container { width: 100%; max-width: 800px; padding: 1rem; flex: 1; }
        .hero { text-align: center; margin-bottom: 3rem; }
        .main-card { background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 2.5rem; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); }
        .upload-area { border: 2px dashed rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 3rem 2rem; text-align: center; cursor: pointer; transition: all 0.3s ease; }
        .upload-area:hover { border-color: var(--primary); background: rgba(16, 185, 129, 0.05); }
        #image-preview { max-width: 100%; max-height: 300px; border-radius: 12px; margin-top: 1rem; display: none; object-fit: cover; }
        .btn { background: var(--primary); color: white; border: none; padding: 1rem 2.5rem; border-radius: 12px; font-weight: 600; cursor: pointer; transition: all 0.3s ease; width: 100%; margin-top: 1.5rem; display: flex; align-items: center; justify-content: center; gap: 10px; }
        .btn:hover { background: var(--primary-dark); transform: translateY(-2px); }
        .result-section { margin-top: 2rem; display: none; animation: fadeInUp 0.5s ease-out; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 2rem; }
        .result-badge { display: inline-block; padding: 0.5rem 1.5rem; border-radius: 50px; font-weight: 700; text-transform: uppercase; margin-bottom: 1rem; }
        .badge-recyclable { background: rgba(56, 189, 248, 0.2); color: var(--accent); }
        .badge-wet { background: rgba(16, 185, 129, 0.2); color: var(--primary); }
        .badge-hazardous { background: rgba(239, 68, 68, 0.2); color: var(--danger); }
        .badge-dry { background: rgba(245, 158, 11, 0.2); color: var(--warning); }
        .confidence-bar-bg { background: rgba(255, 255, 255, 0.05); height: 8px; border-radius: 10px; width: 100%; margin-top: 1rem; overflow: hidden; }
        .confidence-bar { background: var(--primary); height: 100%; width: 0%; transition: width 1s ease; }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <nav><div class="logo">♻️ Eco<span>Sort</span> Portable</div><div>v1.1</div></nav>
    <div class="container">
        <div class="hero"><h1>Waste Classifier AI</h1><p>Single-file portable waste management solution.</p></div>
        <div class="main-card">
            <div class="upload-area" onclick="document.getElementById('file-input').click()">
                <span style="font-size: 3rem;">📸</span><p id="upload-text">Upload Image</p>
                <img id="image-preview" src=""><input type="file" id="file-input" accept="image/*" style="display:none" onchange="previewImage(event)">
            </div>
            <button class="btn" id="predict-btn" onclick="uploadAndPredict()" disabled><span id="btn-text">Analyze Waste</span></button>
            <div class="result-section" id="result-section">
                <div id="result-badge" class="result-badge"></div>
                <p>Detected: <span id="detected-item" style="color: white; font-weight: 600;"></span></p>
                <div id="explanation-box" style="margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                    <p style="font-size: 0.8rem; color: var(--primary); font-weight: 600;">💡 Why this category?</p>
                    <p id="explanation-text" style="font-size: 0.85rem; color: var(--text-muted);"></p>
                </div>
                <div style="margin-top: 1rem; display: flex; justify-content: space-between; font-size: 0.8rem;"><span>Confidence Score</span><span id="confidence-text">0%</span></div>
                <div class="confidence-bar-bg"><div class="confidence-bar" id="confidence-bar"></div></div>
            </div>
        </div>
    </div>
    <script>
        function previewImage(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = e => {
                    document.getElementById('image-preview').src = e.target.result;
                    document.getElementById('image-preview').style.display = 'block';
                    document.getElementById('upload-text').style.display = 'none';
                    document.getElementById('predict-btn').disabled = false;
                };
                reader.readAsDataURL(file);
            }
        }
        async function uploadAndPredict() {
            const fileInput = document.getElementById('file-input');
            const btn = document.getElementById('predict-btn');
            const resultSection = document.getElementById('result-section');
            if (fileInput.files.length === 0) return;
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            btn.disabled = true; document.getElementById('btn-text').innerText = 'Analyzing...';
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const result = await response.json();
                if (result.success) {
                    document.getElementById('detected-item').innerText = result.raw_label;
                    document.getElementById('confidence-text').innerText = (result.confidence * 100).toFixed(1) + '%';
                    document.getElementById('explanation-text').innerText = result.explanation;
                    const badge = document.getElementById('result-badge');
                    badge.innerText = result.prediction;
                    badge.className = 'result-badge ' + 'badge-' + result.prediction.split(' ')[0].toLowerCase();
                    document.getElementById('confidence-bar').style.width = (result.confidence * 100) + '%';
                    resultSection.style.display = 'block';
                }
            } catch (err) { alert('Error: ' + err); }
            finally { btn.disabled = false; document.getElementById('btn-text').innerText = 'Analyze Waste'; }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(BASE_HTML)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file: return jsonify({'success': False, 'error': 'No file'})
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    prediction, confidence, raw_label, explanation = predict_image(filepath)
    return jsonify({'success': True, 'prediction': prediction, 'confidence': confidence, 'raw_label': raw_label, 'explanation': explanation})

def open_browser(): webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(host='0.0.0.0', port=5000)
