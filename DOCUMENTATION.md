# VerifyJob.ai | Project Documentation

VerifyJob.ai is an advanced, AI-powered platform designed to protect job seekers from employment scams and phishing attempts. By analyzing job descriptions using natural language processing (NLP) and machine learning, the system identifies high-risk patterns and provides users with actionable insights.

---

## 🚀 System Overview

The application follows a modern client-server architecture:
- **Frontend**: A high-performance React application built with Vite, styled with Tailwind CSS, and featuring smooth animations via Framer Motion.
- **Backend**: A robust Flask API (Python) that handles text processing, OCR (Optical Character Recognition), and model inference.
- **Intelligence Layer**: A Random Forest ensemble model trained on a comprehensive dataset of real and fraudulent job postings.

---

## 🛠️ Technical Stack

### Frontend
- **Framework**: React.js (Vite)
- **Styling**: Tailwind CSS (Vanilla CSS core)
- **Animations**: Framer Motion
- **Fonts**: Plus Jakarta Sans (Body), Outfit (Headings)
- **Icons**: Google Material Symbols

### Backend
- **Framework**: Flask (Python)
- **NLP**: NLTK (Stopword removal, Lemmatization)
- **OCR**: Tesseract OCR (`pytesseract`)
- **Web Scraping**: BeautifulSoup4, Requests
- **Machine Learning**: Scikit-Learn (Random Forest, TF-IDF Vectorization)

---

## 🧠 Model & Intelligence

### Preprocessing Pipeline
Every input (Text, Image, or URL) is standardized through the following pipeline:
1. **Lowercase Conversion**: Uniformity in feature extraction.
2. **Punctuation Removal**: Filtering out noise.
3. **Lemmatization**: Reducing words to their root form (e.g., "running" -> "run").
4. **Stopword Filtering**: Removing common but insignificant words (the, is, at).

### Classification Model
- **Algorithm**: Random Forest Classifier.
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) to weigh the importance of specific words.
- **Features**: The model analyzes word frequency, structure, and presence of "hotspots" (e.g., requests for money, WhatsApp-only contact).

---

## 🔍 Core Features

### 1. Text Analysis
Users can paste raw job descriptions. the system identifies suspicious phrasing and flags them as "Hotspots".

### 2. Image OCR Extraction
By uploading a screenshot, the system uses Tesseract OCR to extract text content from the image and feed it into the neural engine.

### 3. URL Scraper (Beta)
The system attempts to visit the provided URL, extract the visible text content (ignoring scripts and styles), and analyze it for fraudulent patterns.

---

## 🛣️ Roadmap & Next Steps

### Phase 1: Explainable AI (XAI)
- **Goal**: Don't just show a percentage; show *why*.
- **Plan**: Integrate SHAP or LIME visualizations to highlight exactly which sentences influenced the "Fake" prediction.

### Phase 2: Multilingual Support
- **Goal**: Global protection.
- **Plan**: Expand training data to include Hindi, Spanish, and French job postings to detect regional phishing patterns.

### Phase 3: Browser Extension
- **Goal**: Real-time safety.
- **Plan**: Create a Chrome/Edge extension that automatically scans LinkedIn, Naukri, and Indeed job pages as the user browses.

### Phase 4: Verification Database
- **Goal**: Trusted company registry.
- **Plan**: Cross-reference job postings with a database of verified company domains and official careers pages.

---

## 💻 Local Setup

1. **Backend**:
   ```bash
   pip install flask flask-cors nltk beautifulsoup4 pytesseract pillow scikit-learn
   python app_v2.py
   ```
   *Note: Ensure Tesseract-OCR is installed on your system path.*

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

---

## 🛠️ Troubleshooting (Common Issues)

### OCR Error: "Tesseract is not installed"
If you see an error when uploading images, follow these steps:
1. **Download**: Install the Tesseract binary from [UB-Mannheim's Github](https://github.com/UB-Mannheim/tesseract/wiki).
2. **Install**: Use the default path `C:\Program Files\Tesseract-OCR`.
3. **Verify**: Restart your Python backend. The system will automatically detect the path.

### Backend Port Conflicts
If port `5000` is in use:
- Check for other running Python processes or AirPlay Receiver (on macOS).
- Change `app.run(port=5000)` in `app_v2.py` if necessary.

---

*© 2024 VerifyJob.ai | Integrated AI Analysis*
