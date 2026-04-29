<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini_AI-Powered-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

<h1 align="center">🛡️ HireGuard AI — Fake Job Posting Detector</h1>

<p align="center">
  <b>An AI-powered platform that protects job seekers from employment scams using Machine Learning, NLP, and Google Gemini AI.</b>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-model-performance">Model Performance</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-api-reference">API Reference</a> •
  <a href="#-project-structure">Project Structure</a>
</p>

---

## 📌 Problem Statement

Online job fraud is a growing menace. Scammers exploit desperate job seekers by posting fake listings that demand upfront payments, steal personal data, or impersonate legitimate companies. According to the FBI's Internet Crime Report, job scams cost victims **over $300 million annually**.

**HireGuard AI** tackles this problem by combining a **Random Forest ML model** trained on 18,000+ real-world job postings with **Google Gemini AI** to deliver real-time fraud detection, company verification, and career guidance — all through an intuitive web interface.

---

## ✨ Features

### 🔍 Multi-Input Analysis
| Input Method | Description |
|---|---|
| **📝 Text Analysis** | Paste any job description and get an instant fraud score with detailed reasoning |
| **📸 Image OCR** | Upload a screenshot of a job posting — text is extracted via Gemini Vision (with Tesseract OCR fallback) |
| **🔗 URL Scraper** | Provide a job listing URL — the system scrapes, extracts, and analyzes the content automatically |

### 🧠 Intelligence Engine
- **Hybrid Fraud Scoring** — Weighted combination of ML model probability (30%), Company Trust Score (25%), Heuristic Red Flags (25%), and Gemini AI Risk Rating (20%)
- **Context-Aware Heuristic Engine** — Uses negative lookbehind regex to differentiate between "*requires security deposit*" (🚩) vs "*no security deposit*" (✅)
- **Automatic Company Verification** — Generates a due-diligence report with trust score, red flags, online presence audit, and ownership transparency analysis
- **5-Step Transparent Pipeline** — Every analysis shows the exact steps: Text Preprocessing → TF-IDF Vectorization → Pattern Scan → ML Classification → Feature Importance

### 💬 AI Career Chatbot
- Built-in **HireGuard AI Career Assistant** powered by Gemini
- Provides eligibility analysis, skill gap identification, learning roadmaps, and scam safety advice
- Context-aware — uses the analyzed job posting data to give personalized career guidance
- Graceful fallback with static safety advice when API quota is exceeded

### ⚡ Resilient Architecture
- **Multi-Key API Rotation** — Supports multiple Gemini API keys with automatic fallback across 7+ model variants
- **Graceful Degradation** — Every AI feature has a regex/template fallback, so the app works even without API keys
- **Rate Limit Handling** — Exponential backoff with automatic model pool rotation

---

## 🛠️ Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| **React 18** | UI framework |
| **Vite** | Build tool & dev server |
| **Tailwind CSS** | Utility-first styling |
| **Framer Motion** | Smooth animations & transitions |
| **Lucide React** | Icon library |
| **Plus Jakarta Sans / Outfit** | Typography |

### Backend
| Technology | Purpose |
|---|---|
| **Flask** | REST API server |
| **scikit-learn** | Random Forest classifier + TF-IDF vectorization |
| **NLTK** | Text preprocessing (lemmatization, stopword removal) |
| **Google Gemini AI** | Advanced text analysis, company verification, chatbot |
| **Tesseract OCR** | Fallback image text extraction |
| **BeautifulSoup4** | Web scraping for URL analysis |
| **Pillow** | Image processing |

---

## 🔬 How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
│              (Text / Image / URL)                                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: Text Preprocessing                                     │
│  Lowercase → Remove Punctuation → Lemmatize → Remove Stopwords  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2: TF-IDF Vectorization                                   │
│  Convert cleaned text into numerical feature vectors             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
┌─────────────────────┐ ┌──────────────────────────┐
│  STEP 3: Heuristic  │ │  STEP 4: Random Forest   │
│  Pattern Scan       │ │  Classification          │
│  (Context-Aware)    │ │  (18,000+ samples)       │
│                     │ │                          │
│  • IM platforms     │ │  Prediction: Fake/Real   │
│  • Upfront fees     │ │  Confidence: 0-100%      │
│  • ID/data requests │ │                          │
│  • Unrealistic pay  │ │                          │
│  • Free email usage │ │                          │
└────────┬────────────┘ └──────────┬───────────────┘
         │                         │
         └────────┬────────────────┘
                  ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 5: Gemini AI Enhancement (if available)                    │
│  • Detailed Summary Generation                                   │
│  • Company Name Extraction & Cross-Validation                    │
│  • Company Due-Diligence Report (Trust Score)                    │
│  • Job Details Extraction (Title, Salary, Skills)                │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  FINAL: Weighted Fraud Score Calculation                         │
│                                                                  │
│  Score = (ML × 0.30) + (Company × 0.25) +                       │
│          (Flags × 0.25) + (Gemini × 0.20)                       │
│                                                                  │
│  Output: Fraud Score (0-100%), Verdict, Red Flags,               │
│          Company Report, Job Details, Analysis Pipeline           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

The Random Forest classifier was trained on **18,000+ job postings** (real + augmented fraudulent samples) and evaluated against Logistic Regression and Naive Bayes.

### Model Comparison

| Metric | Random Forest | Logistic Regression | Naive Bayes |
|---|:---:|:---:|:---:|
| **Accuracy** | **98%** | 98% | 97% |
| **Precision** | **100%** | 95% | 100% |
| **Recall** | 89% | **94%** | 81% |
| **F1-Score** | **94.2%** | 94.5% | 89.5% |

> Random Forest was selected for its **100% precision** (zero false positives on legitimate jobs) — critical for a fraud detection system where incorrectly flagging real jobs is unacceptable.

### Confusion Matrix (Test Set: 4,356 samples)

|  | Predicted Real | Predicted Fake |
|---|:---:|:---:|
| **Actual Real** | 3,632 (TN) | 0 (FP) |
| **Actual Fake** | 79 (FN) | 645 (TP) |

### Feature Importance — Top Fraud Indicators

The model identified these words as the strongest fraud signals: `company`, `year`, `team`, `fee`, `spot`, `earn`, `asked`, `upfront`, `home`, `secure`.

---

## 🚀 Installation

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** & npm
- **Tesseract OCR** ([Download from UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)) — *optional, only needed as fallback for image analysis*
- **Google Gemini API Key** ([Get one free](https://aistudio.google.com/apikey)) — *optional, enables AI-enhanced analysis*

### 1. Clone the Repository

```bash
git clone https://github.com/parthpatel16/FakeJobDetector.git
cd FakeJobDetector
```

### 2. Backend Setup

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Download NLTK data (auto-downloads on first run, but can be done manually)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 3. Configure Environment Variables

Create a `.env` file inside the `backend/` directory:

```env
# Single key
GEMINI_API_KEY="your-gemini-api-key-here"

# OR multiple keys for rate-limit resilience (comma-separated)
GEMINI_API_KEYS="key1,key2,key3"
```

> **Note:** The app works without API keys — ML classification and heuristic checks still function. Gemini AI simply adds enhanced summaries, company verification, and the chatbot.

### 4. Frontend Setup

```bash
cd frontend
npm install
```

### 5. Run the Application

Open **two terminals**:

```bash
# Terminal 1: Start Backend (Flask API on port 5000)
cd backend
python main.py

# Terminal 2: Start Frontend (Vite dev server on port 5173)
cd frontend
npm run dev
```

Visit **http://localhost:5173** in your browser.

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `POST /predict-text` | POST | Analyze a job description text |
| `POST /predict-image` | POST | Upload an image for OCR + analysis |
| `POST /predict-url` | POST | Scrape a URL and analyze content |
| `POST /company-verify` | POST | Run company due-diligence check |
| `POST /chat` | POST | Send message to AI career chatbot |
| `GET /chat/status` | GET | Check chatbot availability |
| `GET /health` | GET | API health check |
| `GET /stats` | GET | Session analysis statistics |

### Example Request

```bash
curl -X POST http://localhost:5000/predict-text \
  -H "Content-Type: application/json" \
  -d '{"text": "Urgent hiring! Earn ₹50,000/month from home. No experience needed. Send ₹500 registration fee to start immediately. Contact on WhatsApp: +91-9876543210"}'
```

### Example Response

```json
{
  "prediction": "Fake",
  "fraud_score": 92.5,
  "ml_confidence": 97.3,
  "highlights": ["earn", "fee", "home", "work", "urgent"],
  "company_name": "Unknown",
  "analysis": {
    "summary": "This posting exhibits multiple high-risk fraud indicators...",
    "red_flags": [
      "Communication via WhatsApp instead of official channels",
      "Request for upfront registration fee of ₹500",
      "Unrealistic salary promise with no experience required"
    ],
    "recommendation": "Do not share personal documents or pay any fees."
  },
  "pipeline": [ ... ]
}
```

---

## 📁 Project Structure

```
FakeJobDetector/
├── backend/
│   ├── main.py                  # Flask API server (all routes & logic)
│   ├── requirements.txt         # Python dependencies
│   ├── .env                     # API keys (gitignored)
│   ├── models/
│   │   ├── model.pkl            # Trained Random Forest model
│   │   └── vectorizer.pkl       # TF-IDF vectorizer
│   ├── data/
│   │   ├── fake_job_postings.csv          # Augmented training dataset
│   │   ├── fake_job_postings_original.csv # Original Kaggle dataset
│   │   ├── confusion_matrix.png           # Model evaluation chart
│   │   ├── feature_importance.png         # Top fraud indicators
│   │   ├── model_comparison.png           # Algorithm comparison
│   │   └── compare_models.py              # Model benchmarking script
│   ├── results/
│   │   └── confusion_matrix.png           # Test results
│   └── src/
│       └── legacy/                        # Previous versions
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main React application
│   │   ├── main.jsx             # React entry point
│   │   └── index.css            # Global styles
│   ├── package.json             # Node.js dependencies
│   ├── vite.config.js           # Vite configuration
│   ├── tailwind.config.js       # Tailwind CSS configuration
│   └── postcss.config.js        # PostCSS configuration
│
├── .gitignore
├── DOCUMENTATION.md             # Detailed technical documentation
└── README.md                    # This file
```

---

## 🧪 Dataset

The model was trained on the [Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) dataset from Kaggle, augmented with **thousands of synthetic Indian job scam samples** containing modern fraud patterns:

- WhatsApp/Telegram-only contact
- Registration fee / security deposit demands
- Vague job descriptions with unrealistic pay
- Social media engagement scams ("Comment Interested")
- Cryptocurrency/investment lures

---

## 🛣️ Roadmap

- [x] Multi-input analysis (Text, Image, URL)
- [x] Gemini AI-powered company verification
- [x] AI Career Chatbot with context-aware responses
- [x] Multi-key API rotation with graceful fallback
- [x] Context-aware heuristic engine (negative lookbehind)
- [ ] **Explainable AI (XAI)** — SHAP/LIME visualizations for model transparency
- [ ] **Multilingual Support** — Hindi, Spanish, French fraud detection
- [ ] **Browser Extension** — Real-time scanning on LinkedIn, Naukri, Indeed
- [ ] **Verification Database** — Cross-reference with verified company registries

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Kaggle — Real or Fake Job Posting Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- [Google Gemini AI](https://ai.google.dev/)
- [scikit-learn](https://scikit-learn.org/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Jina AI Reader](https://jina.ai/reader/) — URL content extraction

---

<p align="center">
  <b>Built with ❤️ to protect job seekers from employment fraud</b>
  PPT LINK - https://docs.google.com/presentation/d/1T1tLBFNeaQOLA-PVKQMcmmlbiaqZqjY7/edit?usp=sharing&ouid=104224097209013718232&rtpof=true&sd=true
</p>
