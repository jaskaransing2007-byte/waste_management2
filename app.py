import os
import sqlite3
import datetime
import webbrowser
from threading import Timer
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

# ML Imports (Google Gemini AI - Lightweight & High Accuracy)
import google.generativeai as genai

# --- CONFIGURATION ---
# Use /tmp for Vercel/Serverless environments as the root is read-only
IS_VERCEL = "VERCEL" in os.environ
UPLOAD_FOLDER = '/tmp/uploads' if IS_VERCEL else 'uploads'
DATABASE = '/tmp/waste_management.db' if IS_VERCEL else 'waste_management.db'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Get Gemini API Key from environment or placeholder
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

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
        # Load image
        img = Image.open(img_path)
        
        # Enhanced prompt for Gemini
        prompt = """
        Analyze this image of waste. Categorize it into EXACTLY one of these four categories:
        1. Recyclable Waste (Paper, plastic, glass, metal, cardboard)
        2. Wet Waste (Food scraps, organic matter, vegetable peels)
        3. Hazardous Waste (Batteries, electronics, chemicals, medical waste)
        4. Dry Waste (Non-recyclable paper, dust, cloth, other dry non-recyclables)

        Return only a JSON object with this exact structure:
        {
          "category": "Category Name",
          "detected_item": "Specific name of the item",
          "confidence": 0.95
        }
        """
        
        response = model.generate_content([prompt, img])
        
        # Parse response (Gemini sometimes adds markdown backticks)
        import json
        text_response = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(text_response)
        
        category = data.get("category", "Dry Waste")
        confidence = float(data.get("confidence", 0.90))
        raw_label = data.get("detected_item", "Unknown")
        
        return category, confidence, raw_label
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Dry Waste", 0.5, f"Error: {str(e)}"

# --- FRONTEND TEMPLATE ---
BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EcoSort AI | Vercel Optimized</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #10b981;
            --primary-dark: #059669;
            --bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --danger: #ef4444;
            --warning: #f59e0b;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Outfit', sans-serif;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(56, 189, 248, 0.05) 0%, transparent 40%);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            overflow-x: hidden;
        }

        nav {
            width: 100%;
            padding: 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -1px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .logo span { color: var(--primary); }

        .container {
            width: 100%;
            max-width: 800px;
            padding: 1rem;
            flex: 1;
        }

        .hero {
            text-align: center;
            margin-bottom: 3rem;
            animation: fadeInDown 0.8s ease-out;
        }

        .hero h1 {
            font-size: 3rem;
            margin-bottom: 1rem;
            line-height: 1.1;
        }

        .hero p {
            color: var(--text-muted);
            font-size: 1.1rem;
            max-width: 600px;
            margin: 0 auto;
        }

        .main-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            margin-bottom: 3rem;
            transition: transform 0.3s ease;
        }

        .upload-area {
            border: 2px dashed rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
        }

        .upload-area:hover {
            border-color: var(--primary);
            background: rgba(16, 185, 129, 0.05);
        }

        .upload-area input {
            display: none;
        }

        .upload-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            display: block;
        }

        #image-preview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 12px;
            margin-top: 1rem;
            display: none;
            object-fit: cover;
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        }

        .btn {
            background: var(--primary);
            color: white;
            border: none;
            padding: 1rem 2.5rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .btn:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
            box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.5);
        }

        .btn:disabled {
            opacity: 0.7;
            cursor: not-allowed;
        }

        .result-section {
            margin-top: 2rem;
            display: none;
            animation: fadeInUp 0.5s ease-out;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding-top: 2rem;
        }

        .result-badge {
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 700;
            font-size: 1.2rem;
            margin-bottom: 1rem;
            text-transform: uppercase;
        }

        .badge-recyclable { background: rgba(56, 189, 248, 0.2); color: var(--accent); }
        .badge-wet { background: rgba(16, 185, 129, 0.2); color: var(--primary); }
        .badge-hazardous { background: rgba(239, 68, 68, 0.2); color: var(--danger); }
        .badge-dry { background: rgba(245, 158, 11, 0.2); color: var(--warning); }

        .confidence-bar-bg {
            background: rgba(255, 255, 255, 0.05);
            height: 10px;
            border-radius: 10px;
            width: 100%;
            margin-top: 1rem;
            overflow: hidden;
        }

        .confidence-bar {
            background: var(--primary);
            height: 100%;
            width: 0%;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .history-section {
            width: 100%;
            margin-top: 2rem;
        }

        .history-section h2 {
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .history-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 1.5rem;
        }

        .history-card {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .history-card img {
            width: 100%;
            height: 120px;
            object-fit: cover;
            border-radius: 8px;
        }

        .loading-spinner {
            display: none;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

        @media (max-width: 600px) {
            .hero h1 { font-size: 2rem; }
            .main-card { padding: 1.5rem; }
        }
    </style>
</head>
<body>
    <nav>
        <div class="logo">♻️ Eco<span>Sort</span> AI</div>
        <div>v1.0.0</div>
    </nav>

    <div class="container">
        <div class="hero">
            <h1>Smarter Waste Management</h1>
            <p>Upload an image of your waste item and let our AI classify it instantly for proper disposal.</p>
        </div>

        <div class="main-card">
            <div class="upload-area" onclick="document.getElementById('file-input').click()">
                <span class="upload-icon">📸</span>
                <p id="upload-text">Click or Drop Image Here</p>
                <img id="image-preview" src="" alt="Preview">
                <input type="file" id="file-input" accept="image/*" onchange="previewImage(event)">
            </div>

            <button class="btn" id="predict-btn" onclick="uploadAndPredict()" disabled>
                <div class="loading-spinner" id="spinner"></div>
                <span id="btn-text">Analyze Waste</span>
            </button>

            <div class="result-section" id="result-section">
                <div id="result-badge" class="result-badge">Recyclable</div>
                <p style="color: var(--text-muted); font-size: 0.9rem;">
                    Detected: <span id="detected-item" style="color: white; font-weight: 600;">Plastic Bottle</span>
                </p>
                <div style="margin-top: 1rem; display: flex; justify-content: space-between; font-size: 0.8rem;">
                    <span>Confidence Score</span>
                    <span id="confidence-text">98%</span>
                </div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar" id="confidence-bar"></div>
                </div>

                <div id="feedback-section" style="margin-top: 1.5rem; text-align: center; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 1rem;">
                    <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem;">Was this prediction accurate?</p>
                    <div style="display: flex; gap: 10px; justify-content: center;">
                        <button onclick="sendFeedback('Correct')" class="btn" style="padding: 0.5rem 1rem; margin: 0; font-size: 0.8rem; background: rgba(16, 185, 129, 0.2); color: var(--primary); border: 1px solid var(--primary);">✅ Correct</button>
                        <button onclick="sendFeedback('Incorrect')" class="btn" style="padding: 0.5rem 1rem; margin: 0; font-size: 0.8rem; background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger);">❌ Incorrect</button>
                    </div>
                </div>
            </div>
        </div>

        <div class="history-section">
            <h2>📜 Recent Results</h2>
            <div class="history-grid" id="history-grid">
                <!-- History items will be loaded here -->
            </div>
        </div>
    </div>

    <script>
        function previewImage(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('image-preview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    document.getElementById('upload-text').style.display = 'none';
                    document.getElementById('predict-btn').disabled = false;
                }
                reader.readAsDataURL(file);
            }
        }

        async function loadHistory() {
            const response = await fetch('/history');
            const data = await response.json();
            const grid = document.getElementById('history-grid');
            grid.innerHTML = '';
            
            data.forEach(item => {
                const card = document.createElement('div');
                card.className = 'history-card';
                const badgeClass = 'badge-' + item.prediction.split(' ')[0].toLowerCase();
                card.innerHTML = `
                    <div class="result-badge ${badgeClass}" style="font-size: 0.8rem; padding: 4px 10px;">${item.prediction}</div>
                    <p style="font-size: 0.8rem; color: var(--text-muted)">${item.timestamp}</p>
                `;
                grid.appendChild(card);
            });
        }

        async function uploadAndPredict() {
            const fileInput = document.getElementById('file-input');
            const btn = document.getElementById('predict-btn');
            const spinner = document.getElementById('spinner');
            const btnText = document.getElementById('btn-text');
            const resultSection = document.getElementById('result-section');

            if (fileInput.files.length === 0) return;

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            btn.disabled = true;
            spinner.style.display = 'block';
            btnText.innerText = 'Analyzing...';
            resultSection.style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                if (result.success) {
                    document.getElementById('detected-item').innerText = result.raw_label;
                    document.getElementById('confidence-text').innerText = (result.confidence * 100).toFixed(1) + '%';
                    
                    const badge = document.getElementById('result-badge');
                    badge.innerText = result.prediction;
                    badge.className = 'result-badge ' + 'badge-' + result.prediction.split(' ')[0].toLowerCase();
                    
                    document.getElementById('confidence-bar').style.width = (result.confidence * 100) + '%';
                    
                    resultSection.style.display = 'block';
                    loadHistory();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (err) {
                alert('Connection Error');
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
                btnText.innerText = 'Analyze Waste';
            }
        }

        async function sendFeedback(type) {
            const feedbackSection = document.getElementById('feedback-section');
            // In a real app, you'd send the last ID, but here we'll just show a thank you
            feedbackSection.innerHTML = `<p style="color: var(--primary); font-size: 0.9rem;">Thank you for your feedback! 🌿</p>`;
            
            // Optional: Send to backend
            try {
                await fetch('/feedback', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({feedback: type})
                });
            } catch (e) {}
        }

        // Initialize
        loadHistory();
    </script>
</body>
</html>
"""

# --- ROUTES ---

@app.route('/')
def index():
    return render_template_string(BASE_HTML)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        # Add timestamp to filename to avoid collisions
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # ML Prediction
        prediction, confidence, raw_label = predict_image(filepath)
        
        if prediction:
            # Save to DB
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(DATABASE)
            c = conn.cursor()
            c.execute("INSERT INTO predictions (filename, prediction, raw_label, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
                      (filename, prediction, raw_label, confidence, timestamp))
            conn.commit()
            conn.close()

            return jsonify({
                'success': True,
                'prediction': prediction,
                'confidence': confidence,
                'raw_label': raw_label,
                'timestamp': timestamp
            })
        else:
            return jsonify({'success': False, 'error': 'Prediction failed'})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    feedback_text = data.get('feedback')
    
    # Update the latest entry with feedback
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("UPDATE predictions SET user_feedback = ? WHERE id = (SELECT MAX(id) FROM predictions)", (feedback_text,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/history')
def history():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 5")
    rows = c.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            'prediction': row['prediction'],
            'timestamp': row['timestamp']
        })
    return jsonify(results)

def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    # Automatically open browser after 1 second
    Timer(1, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
