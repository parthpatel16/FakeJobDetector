import os
import re
import pickle
import json
import numpy as np
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import requests
import pytesseract
from PIL import Image
import io
import base64
from dotenv import load_dotenv
import google.generativeai as genai
import traceback

# Load environment variables
load_dotenv(override=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# --- Configuration ---
# Point to Tesseract executable for OCR on Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Gemini AI Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_model = None
gemini_model_name = None

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"[INFO] API key loaded ({GEMINI_API_KEY[:10]}...)")
    
    # Models in order of preference (correct names from API)
    model_candidates = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",  
        "gemini-1.5-flash",
    ]
    
    for name in model_candidates:
        try:
            test_model = genai.GenerativeModel(name)
            # Verify it actually works with a quick call
            test_model.generate_content("Hi", generation_config={"max_output_tokens": 5})
            gemini_model = test_model
            gemini_model_name = name
            print(f"[INFO] Gemini AI verified and ready: {name}")
            break
        except Exception as e:
            print(f"[WARN] Model '{name}' failed: {e}")
            continue
    
    if not gemini_model:
        print("[ERROR] All Gemini models failed. Chatbot disabled.")
else:
    print("[WARN] GEMINI_API_KEY not set. Chatbot will be disabled.")

# --- Load Model and Vectorizer ---
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("[INFO] Model and Vectorizer loaded.")
except Exception as e:
    print(f"[ERROR] Error loading models: {e}")

# --- NLTK Setup ---
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Standardized preprocessing matching the training phase."""
    if not text: return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in STOP_WORDS])
    return text

# --- Company & Job Details Extraction ---

def extract_company_name(text):
    """Extracts probable company name from text using multiple heuristic patterns.
    This is the FALLBACK method when Gemini is unavailable."""
    patterns = [
        r"(?:company|employer|organization|firm|agency)\s*(?:name)?\s*[:\-]\s*(.+?)(?:\n|\.|,|$)",
        r"(?:at|@)\s+([A-Z][A-Za-z&\.\s]+(?:Ltd|Inc|Corp|LLC|Pvt|Technologies|Solutions|Services|Group|Global|Consulting|Systems|Networks|Labs|Studio|Media|Digital|Software|Enterprises|Industries|International|Associates|Partners|Co\.?))",
        r"([A-Z][A-Za-z&\.\s]+(?:Ltd|Inc|Corp|LLC|Pvt|Technologies|Solutions|Services|Group|Global|Consulting|Systems|Networks|Labs|Studio|Media|Digital|Software|Enterprises|Industries|International|Associates|Partners|Co\.?))",
        r"(?:about|join|work\s+(?:at|for|with))\s+([A-Z][A-Za-z\s&\.]+?)(?:\.|,|\n|!|\?|$)",
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+is\s+(?:hiring|looking|seeking|recruiting)",
    ]
    
    # Extensive skip list to prevent false positives like "years co" from "2 years"
    skip_words = [
        "the", "we", "our", "this", "that", "you", "your", "apply", "click",
        "job", "position", "role", "opportunity", "work", "home", "earn",
        "send", "contact", "call", "email", "resume", "hiring",
        "years", "year", "experience", "salary", "location", "remote",
        "description", "requirements", "qualifications", "skills",
        "full", "part", "time", "contract", "intern", "internship",
        "monday", "tuesday", "wednesday", "thursday", "friday",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "about", "join", "looking", "seeking", "need", "want",
        "good", "great", "best", "top", "new", "immediate",
        "per", "month", "week", "day", "hour", "annual", "annum",
        "lpa", "ctc", "inr", "usd", "package", "compensation",
        "onsite", "on-site", "hybrid", "flexible",
        "years co", "years inc", "years ltd",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            name = match.strip().rstrip(".,;:!?")
            if name and len(name) > 2 and name.lower() not in skip_words:
                # Additional check: skip if the name is just a number + word
                if re.match(r'^\d+\s*\w+$', name):
                    continue
                return name
    
    return None


def extract_company_name_with_gemini(text):
    """Uses Gemini AI to extract the company name from job posting text.
    Cross-validates with regex to prevent truncated/hallucinated names."""
    # Always get the regex result as baseline
    regex_name = extract_company_name(text)
    
    if not gemini_model:
        return regex_name
    
    try:
        prompt = f"""You are a data extraction tool. From the following job posting text, identify the company or organization name that is hiring.

Instructions:
- Return ONLY the company/organization name as plain text, exactly as it appears in the text.
- If the text says "company name: XYZ" or "company: XYZ", return exactly "XYZ" as written.
- Do NOT shorten, abbreviate, or modify the company name.
- If no company or organization name is clearly mentioned, return exactly: Unknown
- Do not return generic words like "years", "experience", "salary", "remote", "developer", etc.

Text:
\"\"\"
{text[:1500]}
\"\"\"

Company name:"""
        
        response = gemini_model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 50, "temperature": 0.0}
        )
        
        # Handle safety filter blocks
        if not response.candidates or not response.candidates[0].content.parts:
            print("[WARN] Gemini returned empty response for company name, using regex fallback")
            return regex_name
        
        gemini_name = response.text.strip().strip('"').strip("'").strip()
        
        # Validate the result
        if not gemini_name or gemini_name.lower() in ["unknown", "n/a", "not found", "not mentioned", "none", "not specified"]:
            return regex_name  # Fall back to regex if Gemini says unknown
        
        # Cross-validate: if regex found a longer name that contains the Gemini name, prefer regex
        # This prevents truncation issues (e.g., Gemini returns "Infos" but regex found "Infosense")
        if regex_name:
            if gemini_name.lower() in regex_name.lower() and len(regex_name) > len(gemini_name):
                print(f"[INFO] Using regex name '{regex_name}' over Gemini name '{gemini_name}' (more complete)")
                return regex_name
        
        # Also verify the Gemini name actually appears in the text (case-insensitive)
        if gemini_name.lower() not in text.lower():
            print(f"[WARN] Gemini name '{gemini_name}' not found in text, using regex fallback")
            return regex_name if regex_name else gemini_name
        
        return gemini_name
    except Exception as e:
        print(f"[WARN] Gemini company name extraction failed: {e}")
        return regex_name


def check_website_status(url):
    """Checks if a website URL is accessible."""
    try:
        resp = requests.head(url, timeout=5, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        return {"url": url, "status": "live" if resp.status_code < 400 else "error", "status_code": resp.status_code}
    except requests.exceptions.ConnectionError:
        return {"url": url, "status": "down", "status_code": None}
    except requests.exceptions.Timeout:
        return {"url": url, "status": "timeout", "status_code": None}
    except Exception:
        return {"url": url, "status": "error", "status_code": None}


def verify_company_with_gemini(company_name, job_text):
    """Generates a comprehensive company due-diligence report using Gemini AI."""
    if not gemini_model or not company_name:
        return None
    
    # Check for website URLs in the text
    url_match = re.findall(r'https?://[^\s<>"]+', job_text)
    website_status = None
    if url_match:
        website_status = check_website_status(url_match[0])
    
    try:
        prompt = f"""You are a corporate due-diligence analyst. Analyze the company "{company_name}" mentioned in this job posting and provide a comprehensive verification report.

Job posting text for context:
---
{job_text[:2000]}
---

You MUST respond in valid JSON format only. No markdown, no extra text. Use this exact structure:
{{
  "name": "{company_name}",
  "location": "City, Country or Unknown",
  "industry": "Industry type or Unknown",
  "business_type": "Brief description of what the company does",
  "website": "Company website URL if known, or Unknown",
  "founded": "Year or estimated range or Unknown",
  "size": "Employee count estimate (e.g., '2-50', '50-200', '200-1000', '1000-5000', '5000+') or Unknown",
  "ownership": "Founder/CEO name if known, or 'Not disclosed'",
  "registration_type": "e.g., Private Limited, LLC, Free Zone, etc. or Unknown",
  "online_presence": {{
    "website_exists": true or false,
    "linkedin_presence": "Description of LinkedIn presence or Unknown",
    "social_media": "Description of social media presence or Unknown"
  }},
  "website_analysis": {{
    "team_shown": true or false or "unknown",
    "portfolio_proof": true or false or "unknown",
    "registration_number_shown": true or false or "unknown",
    "generic_template": true or false or "unknown"
  }},
  "trust_score": 0-100 (integer, be realistic and conservative),
  "trust_breakdown": [
    {{
      "factor": "Factor name",
      "status": "pass" or "fail" or "warn",
      "detail": "Brief explanation"
    }}
  ],
  "red_flags": ["List of specific concerns"],
  "positive_signs": ["List of positive indicators"],
  "verdict": "One-line summary verdict",
  "detailed_report": "A 3-5 paragraph detailed analysis in plain text covering: company overview, online presence audit, ownership transparency, business consistency, and final assessment. Be specific and factual. If you don't have information, say so clearly rather than making assumptions."
}}

IMPORTANT RULES:
1. Be HONEST. If you don't have data about this company, say "No verified data available" rather than making things up.
2. For unknown/small companies, the trust score should be LOW (20-40) unless there's strong evidence.
3. For well-known companies (Google, Microsoft, TCS, Infosys etc.), the trust score should be HIGH (70-95).
4. Include at least 5 factors in trust_breakdown covering: Website, Physical Address, Owner Transparency, Employee Presence, Social Proof, Business Consistency, Registration Proof.
5. The response MUST be valid JSON."""
        
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 4000, 
                "temperature": 0.3,
                "response_mime_type": "application/json"
            }
        )
        
        # Parse the JSON response with robust extraction
        response_text = response.text.strip()
        
        # Strategy 1: Try direct JSON parse
        company_data = None
        try:
            company_data = json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract from markdown code fences
        if company_data is None:
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    company_data = json.loads(json_match.group(1).strip())
                except json.JSONDecodeError:
                    pass
        
        # Strategy 3: Find the first { to last } 
        if company_data is None:
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                try:
                    company_data = json.loads(response_text[first_brace:last_brace + 1])
                except json.JSONDecodeError:
                    pass
        
        if company_data is None:
            raise json.JSONDecodeError("All parsing strategies failed", response_text[:200], 0)
        
        # Add website check result if we performed one
        if website_status:
            company_data["website_check"] = website_status
        
        company_data["source"] = "gemini"
        return company_data
        
    except json.JSONDecodeError as e:
        print(f"[WARN] Gemini company verification returned invalid JSON: {e}")
        print(f"[DEBUG] Raw response: {response.text[:500] if response else 'No response'}")
        return {
            "name": company_name,
            "trust_score": None,
            "verdict": "Verification failed — could not parse AI response",
            "source": "error"
        }
    except Exception as e:
        print(f"[WARN] Gemini company verification failed: {e}")
        traceback.print_exc()
        return {
            "name": company_name,
            "trust_score": None,
            "verdict": "Verification unavailable",
            "source": "error"
        }

def extract_job_details(text):
    """Extracts structured job details from raw text using regex heuristics."""
    low = text.lower()
    details = {}
    
    # Job Title
    title_patterns = [
        r"(?:job\s*title|position|role|designation|opening)\s*[:\-]\s*(.+?)(?:\n|\.|$)",
        r"(?:hiring|looking\s+for|seeking|we\s+need)\s+(?:a\s+)?(.+?)(?:\n|\.|,|$)",
    ]
    for pat in title_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            details["title"] = m.group(1).strip()[:80]
            break
    
    # Location
    loc_patterns = [
        r"(?:location|city|place|based\s+(?:in|at)|office)\s*[:\-]\s*(.+?)(?:\n|\.|,|$)",
        r"(?:remote|work\s+from\s+home|hybrid|on-?site|wfh)",
    ]
    m = re.search(loc_patterns[0], text, re.IGNORECASE)
    if m:
        details["location"] = m.group(1).strip()[:60]
    elif re.search(loc_patterns[1], low):
        match = re.search(loc_patterns[1], low)
        details["location"] = match.group(0).title()
    
    # Salary
    salary_patterns = [
        r"(?:salary|pay|compensation|ctc|stipend|earning|package)\s*[:\-]\s*(.+?)(?:\n|\.|$)",
        r"(?:[\$\u20b9\u00a3\u20ac])\s*[\d,]+(?:\s*[-\/]\s*[\$\u20b9\u00a3\u20ac]?\s*[\d,]+)?(?:\s*(?:per|\/)\s*(?:month|year|week|day|hr|hour|annum))?",
        r"(?:INR|USD|Rs\.?)\s*[\d,]+(?:\s*[-\/]\s*[\d,]+)?",
    ]
    m = re.search(salary_patterns[0], text, re.IGNORECASE)
    if m:
        details["salary"] = m.group(1).strip()[:60]
    else:
        m = re.search(salary_patterns[1], text)
        if m:
            details["salary"] = m.group(0).strip()
        else:
            m = re.search(salary_patterns[2], text, re.IGNORECASE)
            if m:
                details["salary"] = m.group(0).strip()
    
    # Employment Type
    type_keywords = {
        "full time": "Full Time", "full-time": "Full Time", "fulltime": "Full Time",
        "part time": "Part Time", "part-time": "Part Time", "parttime": "Part Time",
        "contract": "Contract", "freelance": "Freelance", "internship": "Internship",
        "temporary": "Temporary", "remote": "Remote", "work from home": "Remote / WFH",
    }
    for keyword, label in type_keywords.items():
        if keyword in low:
            details["employment_type"] = label
            break
    
    # Experience
    exp_patterns = [
        r"(?:experience|exp)\s*[:\-]\s*(.+?)(?:\n|\.|$)",
        r"(\d+[\+]?\s*(?:[-\u2013]\s*\d+\s*)?(?:years?|yrs?))\s+(?:of\s+)?(?:experience|exp)",
    ]
    for pat in exp_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            details["experience"] = m.group(1).strip()[:50]
            break
    
    # Contact Info
    contacts = []
    phone = re.findall(r"[\+]?\d[\d\s\-]{8,14}\d", text)
    if phone:
        contacts.extend([p.strip() for p in phone[:2]])
    email = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    if email:
        contacts.extend(email[:2])
    if contacts:
        details["contact"] = ", ".join(contacts)
    
    # Skills / Requirements (extract keywords)
    skills_section = re.search(
        r"(?:requirements?|qualifications?|skills?\s*(?:required)?|must\s+have|what\s+we.?re\s+looking)\s*[:\-]?\s*(.+?)(?:\n\n|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if skills_section:
        raw_skills = skills_section.group(1)
        # Extract bullet points or comma-separated items
        skill_items = re.findall(r"[-•*]\s*(.+?)(?:\n|$)", raw_skills)
        if not skill_items:
            skill_items = [s.strip() for s in raw_skills.split(",") if s.strip()]
        details["skills"] = [s.strip()[:80] for s in skill_items[:8]]
    
    return details

def get_analysis_pipeline(text, cleaned_text, prediction, confidence, highlights, red_flags):
    """Documents every check performed during the analysis for transparency."""
    low = text.lower()
    word_count = len(text.split())
    cleaned_word_count = len(cleaned_text.split())
    
    pipeline = []
    
    # Step 1: Text Preprocessing
    pipeline.append({
        "step": 1,
        "name": "Text Preprocessing",
        "icon": "text_fields",
        "description": f"Converted {word_count} words to lowercase, removed punctuation and special characters, then performed lemmatization and stopword filtering.",
        "detail": f"{word_count} words in -> {cleaned_word_count} meaningful tokens out",
        "status": "done"
    })
    
    # Step 2: TF-IDF Vectorization
    pipeline.append({
        "step": 2,
        "name": "TF-IDF Vectorization",
        "icon": "data_array",
        "description": "Transformed the cleaned text into numerical feature vectors using Term Frequency-Inverse Document Frequency weighting.",
        "detail": f"Generated feature vector with {len(highlights)} high-impact features identified",
        "status": "done"
    })
    
    # Step 3: Pattern Scanning
    checks_passed = []
    checks_failed = []
    
    im_check = any(w in low for w in ["whatsapp", "telegram", "imo", "messenger"])
    if im_check:
        checks_failed.append("IM platform contact detected")
    else:
        checks_passed.append("No suspicious contact methods")
    
    fee_check = any(w in low for w in ["deposit", "registration fee", "processing fee", "security fee", "refundable"])
    if fee_check:
        checks_failed.append("Upfront payment request detected")
    else:
        checks_passed.append("No upfront fees requested")
    
    id_check = any(w in low for w in ["bank account", "aadhaar", "pan card", "cvv", "otp"])
    if id_check:
        checks_failed.append("Sensitive data request detected")
    else:
        checks_passed.append("No early ID/financial data requests")
    
    scam_check = any(w in low for w in ["daily pay", "earn fast", "no interview", "immediate join", "no experience"])
    if scam_check:
        checks_failed.append("'Too good to be true' language detected")
    else:
        checks_passed.append("No unrealistic promises found")
    
    email_check = bool(re.findall(r"[a-zA-Z0-9_.+-]+@(?:gmail|yahoo|hotmail|outlook)\.\w+", low))
    if email_check:
        checks_failed.append("Free email domain used (not corporate)")
    else:
        checks_passed.append("No suspicious email domains")
    
    pipeline.append({
        "step": 3,
        "name": "Heuristic Pattern Scan",
        "icon": "security",
        "description": f"Scanned text against 5 known fraud pattern categories: IM platforms, upfront fees, ID requests, unrealistic claims, email domains.",
        "detail": f"{len(checks_passed)} passed, {len(checks_failed)} flagged",
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "status": "done"
    })
    
    # Step 4: ML Classification
    pipeline.append({
        "step": 4,
        "name": "Random Forest Classification",
        "icon": "model_training",
        "description": "Fed the TF-IDF vector into a Random Forest ensemble of decision trees trained on 18,000+ job postings.",
        "detail": f"Prediction: {'FRAUDULENT' if prediction == 1 else 'LEGITIMATE'} with {round(confidence * 100, 2)}% confidence",
        "status": "done"
    })
    
    # Step 5: Feature Importance
    pipeline.append({
        "step": 5,
        "name": "Feature Importance Analysis",
        "icon": "insights",
        "description": "Identified the top words from your input that had the highest influence on the model's decision.",
        "detail": f"Top linguistic hotspots: {', '.join(highlights[:5]) if highlights else 'N/A'}",
        "status": "done"
    })
    
    return pipeline

def get_highlights(text, top_n=5):
    """Identifies words that are structurally important in our classification model."""
    cleaned = clean_text(text)
    words = list(set(cleaned.split()))
    if not words: return []

    feature_names = vectorizer.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        weights = model.feature_importances_
    else:
        weights = model.coef_[0]
    
    feature_weight_map = {name: weight for name, weight in zip(feature_names, weights)}
    
    word_scores = []
    for w in words:
        if w in feature_weight_map:
            score = feature_weight_map[w]
            word_scores.append((w, score))
    
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [w for w, s in word_scores[:top_n]]

def get_detailed_analysis(text, prediction, confidence):
    """Generates a detailed summary and reasoning for the result."""
    is_fake = prediction == 1
    low_text = text.lower()
    
    red_flags = []
    if any(word in low_text for word in ["whatsapp", "telegram", "imo", "messenger"]):
        red_flags.append("Communication via instant messaging apps instead of official channels.")
    if any(word in low_text for word in ["deposit", "registration fee", "processing fee", "security fee", "refundable"]):
        red_flags.append("Request for upfront payment or 'security deposits'.")
    if any(word in low_text for word in ["bank account", "aadhaar", "pan card", "cvv", "otp"]):
        red_flags.append("Requests for sensitive personal or financial identification early in the process.")
    if any(word in low_text for word in ["daily pay", "earn fast", "no interview", "immediate join"]):
        red_flags.append("Signs of 'too good to be true' offers or bypassing standard hiring filters.")
    
    summary = ""
    if is_fake:
        summary = "Our analysis indicates high-risk indicators commonly found in employment scams. This includes suspicious phrasing, unprofessional contact methods, or requests for upfront payments."
    else:
        summary = "This posting follows standard corporate recruitment patterns. The language is professional, provides clear requirements, and does not ask for sensitive information upfront."

    company_insight = "Based on the profile, the hiring entity appears to be focusing on "
    if "data entry" in low_text or "typing" in low_text:
        company_insight += "high-volume clerical tasks which are often exploited by scammers."
    elif "engineer" in low_text or "developer" in low_text:
        company_insight += "specialized technical roles typical of established IT services."
    else:
        company_insight += "general operations. Always verify the company URL and official email domain (@company.com)."

    return {
        "summary": summary,
        "red_flags": red_flags if is_fake else [],
        "company_insight": company_insight,
        "recommendation": "Do not share personal documents or pay any fees. Verify on the official company website." if is_fake else "Proceed with standard professional caution."
    }

def get_prediction_data(text):
    """Helper to run prediction and structure the full response."""
    cleaned = clean_text(text)
    if not cleaned:
        return {"error": "No usable text found for analysis."}

    vec = vectorizer.transform([cleaned])
    prediction = int(model.predict(vec)[0])
    probs = model.predict_proba(vec)[0]
    confidence = float(max(probs))

    highlights = get_highlights(text)
    analysis = get_detailed_analysis(text, prediction, confidence)
    
    # Extract company name (use Gemini if available, else fallback to regex)
    company_name = extract_company_name_with_gemini(text)
    job_details = extract_job_details(text)
    
    # Build analysis pipeline
    pipeline = get_analysis_pipeline(
        text, cleaned, prediction, confidence, highlights, analysis.get("red_flags", [])
    )
    
    return {
        "prediction": "Fake" if prediction == 1 else "Real",
        "confidence": round(confidence * 100, 2),
        "highlights": highlights,
        "analysis": analysis,
        "extracted_text": text[:500] + "..." if len(text) > 500 else text,
        "company_name": company_name,
        "job_details": job_details,
        "pipeline": pipeline,
    }

# --- Endpoints ---

@app.route("/predict-text", methods=["POST"])
def predict_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    result = get_prediction_data(data["text"])
    return jsonify(result)

@app.route("/predict-image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        # Build base64 preview for frontend
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime = file.content_type or "image/png"
        
        # Extract text using OCR
        extracted_text = pytesseract.image_to_string(img)
        if not extracted_text.strip():
            return jsonify({"error": "Could not extract any text from the image."}), 400
        
        result = get_prediction_data(extracted_text)
        result["image_preview"] = f"data:{mime};base64,{img_b64}"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500

@app.route("/predict-url", methods=["POST"])
def predict_url():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "No URL provided"}), 400
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(data["url"], headers=headers, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.content, "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean up whitespace
        text = soup.get_text(separator=" ")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)
        
        if not text.strip():
            return jsonify({"error": "Could not extract text from the webpage."}), 400
            
        result = get_prediction_data(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Web scraping failed: {str(e)}"}), 500

# --- Company Verification Endpoint ---

@app.route("/company-verify", methods=["POST"])
def company_verify():
    """Runs a comprehensive company due-diligence check using Gemini AI."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    company_name = data.get("company_name")
    job_text = data.get("job_text", "")
    
    if not company_name:
        return jsonify({
            "company_info": {
                "name": "Unknown",
                "trust_score": 0,
                "verdict": "No company name detected in the job posting. This is a common trait of fraudulent listings.",
                "red_flags": ["No company name provided — legitimate employers always identify themselves."],
                "trust_breakdown": [
                    {"factor": "Company Identity", "status": "fail", "detail": "No company name found in posting"}
                ],
                "source": "none"
            }
        })
    
    if not gemini_model:
        return jsonify({
            "company_info": {
                "name": company_name,
                "trust_score": None,
                "verdict": "AI verification unavailable — Gemini API not configured.",
                "source": "unavailable"
            }
        })
    
    result = verify_company_with_gemini(company_name, job_text)
    
    return jsonify({"company_info": result})

# --- AI Chatbot Endpoint ---

CHATBOT_SYSTEM_PROMPT = """You are VerifyJob.ai Career Assistant — an expert AI career counselor embedded in a job fraud detection platform.

YOUR CAPABILITIES:
1. **Eligibility Analysis**: Compare the user's profile/skills against job requirements and give a clear eligibility breakdown.
2. **Skill Gap Analysis**: Identify exactly which skills the user is missing for a role.
3. **Learning Roadmap**: Generate structured, week-by-week or month-by-month learning plans with free/paid resources (courses, projects, certifications).
4. **Career Guidance**: Resume tips, interview prep, salary negotiation, career switching advice.
5. **Job Safety Advice**: Warn about scam red flags, how to verify companies, safe application practices.

FORMATTING RULES:
- Use markdown formatting: **bold**, *italic*, bullet points, numbered lists.
- Keep responses concise but actionable. Use headers for long responses.
- When giving a roadmap, structure it as a clear timeline with specific resources.
- Always be encouraging but honest about skill gaps.
- If the job was flagged as FAKE, strongly advise against applying and explain why.

CONTEXT: You have access to the analyzed job posting details below. Use this context to give personalized advice.
"""

@app.route("/chat", methods=["POST"])
def chat():
    if not gemini_model:
        return jsonify({"error": "AI Chatbot is not configured. Please set GEMINI_API_KEY in .env file."}), 503
    
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_message = data["message"]
    job_context = data.get("job_context", None)
    chat_history = data.get("history", [])
    
    # Build the context-aware prompt
    system_context = CHATBOT_SYSTEM_PROMPT
    
    if job_context:
        system_context += f"""
--- ANALYZED JOB POSTING ---
Prediction: {job_context.get('prediction', 'N/A')} (Confidence: {job_context.get('confidence', 'N/A')}%)
Company: {job_context.get('company_name', 'Unknown')}
Job Title: {job_context.get('job_details', {}).get('title', 'Not specified')}
Location: {job_context.get('job_details', {}).get('location', 'Not specified')}
Salary: {job_context.get('job_details', {}).get('salary', 'Not specified')}
Experience Required: {job_context.get('job_details', {}).get('experience', 'Not specified')}
Employment Type: {job_context.get('job_details', {}).get('employment_type', 'Not specified')}
Skills Required: {', '.join(job_context.get('job_details', {}).get('skills', [])) if job_context.get('job_details', {}).get('skills') else 'Not specified'}
Original Text Snippet: {job_context.get('extracted_text', '')[:300]}
Red Flags Found: {', '.join(job_context.get('analysis', {}).get('red_flags', [])) if job_context.get('analysis', {}).get('red_flags') else 'None'}
---
"""
    
    try:
        # Build conversation history for Gemini
        gemini_history = []
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})
        
        # Start chat with history
        chat = gemini_model.start_chat(history=gemini_history)
        
        # Send current message with system context prepended to first message
        if not gemini_history:
            full_message = f"{system_context}\n\nUser: {user_message}"
        else:
            full_message = user_message
        
        response = chat.send_message(full_message)
        
        return jsonify({
            "reply": response.text,
            "status": "ok"
        })
    except Exception as e:
        return jsonify({"error": f"AI response failed: {str(e)}"}), 500

@app.route("/chat/status", methods=["GET"])
def chat_status():
    """Check if the chatbot is available."""
    return jsonify({
        "available": gemini_model is not None,
        "model": gemini_model_name if gemini_model else None
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
