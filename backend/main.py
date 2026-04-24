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
# --- Load Environment Variables ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
print(f"[INFO] Environment loaded from: {os.path.join(BASE_DIR, '.env')}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# --- Configuration ---
# Point to Tesseract executable for OCR on Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Gemini AI Setup (Multi-Key + Multi-Model Pool) ---
import time as _time

gemini_model = None
gemini_model_name = None
gemini_models_pool = []  # All available key+model combos for rate-limit fallback

# Collect all API keys: GEMINI_API_KEYS (comma-separated) takes priority, 
# falls back to single GEMINI_API_KEY
_raw_keys = os.getenv("GEMINI_API_KEYS", "")
_single_key = os.getenv("GEMINI_API_KEY", "")

api_keys = []
if _raw_keys:
    api_keys = [k.strip() for k in _raw_keys.split(",") if k.strip() and k.strip() != "YOUR_API_KEY_HERE"]
if not api_keys and _single_key and _single_key != "YOUR_API_KEY_HERE":
    api_keys = [_single_key]

if api_keys:
    print(f"[INFO] {len(api_keys)} API key(s) loaded")
    
    # Models in order of preference
    model_candidates = [
        "gemini-flash-latest",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.5-pro-latest",
    ]
    
    # Register all key × model combinations
    # Each API key has its own independent quota, so this multiplies available requests
    for key_index, api_key in enumerate(api_keys):
        key_label = f"Key{key_index + 1}({api_key[:8]}...)"
        try:
            genai.configure(api_key=api_key)
            for name in model_candidates:
                try:
                    model_obj = genai.GenerativeModel(name)
                    gemini_models_pool.append({
                        "name": name,
                        "model": model_obj,
                        "api_key": api_key,
                        "key_label": key_label
                    })
                except Exception as e:
                    pass  # silently skip unavailable models
            print(f"[INFO] {key_label}: models registered")
        except Exception as e:
            print(f"[WARN] {key_label}: configuration failed — {e}")
    
    if gemini_models_pool:
        # Configure with the first key as default
        genai.configure(api_key=gemini_models_pool[0]["api_key"])
        gemini_model = gemini_models_pool[0]["model"]
        gemini_model_name = gemini_models_pool[0]["name"]
        total_models = len(gemini_models_pool)
        print(f"[INFO] Primary: {gemini_model_name} | {total_models} model(s) across {len(api_keys)} key(s)")
    else:
        print("[ERROR] All Gemini models failed across all keys. AI features disabled.")
else:
    print("[WARN] No GEMINI_API_KEYS or GEMINI_API_KEY set. AI features disabled.")


# Track which API key is currently configured to avoid unnecessary reconfiguration
_current_api_key = gemini_models_pool[0]["api_key"] if gemini_models_pool else None

def safe_generate(prompt, generation_config=None, max_retries=2, contents=None):
    """Generates content using Gemini with automatic key + model fallback.
    Supports multimodal input (text + images)."""
    global _current_api_key
    
    if not gemini_models_pool:
        return None
    
    # If contents is provided, use it (multimodal), otherwise wrap prompt in list
    input_data = contents if contents is not None else [prompt]
    
    last_error = None
    for model_info in gemini_models_pool:
        if model_info["api_key"] != _current_api_key:
            genai.configure(api_key=model_info["api_key"])
            _current_api_key = model_info["api_key"]
        
        for attempt in range(max_retries):
            try:
                response = model_info["model"].generate_content(
                    input_data,
                    generation_config=generation_config or {}
                )
                return response
            except Exception as e:
                error_str = str(e)
                last_error = e
                if "429" in error_str or "ResourceExhausted" in error_str:
                    if attempt < max_retries - 1:
                        _time.sleep(2 ** attempt)
                    else:
                        break
                else:
                    break
    return None

# --- Load Model and Vectorizer ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(BASE_DIR, "models", "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models", "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    print("[INFO] Model and Vectorizer loaded.")
except Exception as e:
    print(f"[ERROR] Error loading models: {e}")
    model = None
    vectorizer = None

# --- Gemini API Check ---
keys = os.getenv("GEMINI_API_KEYS", "").split(",")
if keys and keys[0].strip():
    print(f"[INFO] Found {len(keys)} Gemini API keys in .env")
else:
    print("[WARN] No Gemini API keys found! AI features will be disabled.")

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
    
    if not gemini_models_pool:
        return regex_name
    
    try:
        prompt = f"""You are a professional data analyst. Identify the hiring company/organization from this job posting.
        
Instructions:
- Return ONLY the official company name.
- EXCLUDE locations (e.g., if it says 'TCS Kolkata', return 'TCS').
- EXCLUDE job titles (e.g., if it says 'Pega Developer - Infosys', return 'Infosys').
- Do NOT include generic filler words.
- If unsure or not mentioned, return exactly: Unknown

Text:
\"\"\"
{text[:40000]}
\"\"\"

Company name:"""
        
        response = safe_generate(prompt, generation_config={"max_output_tokens": 50, "temperature": 0.0})
        
        # Handle empty/failed response
        if not response or not response.candidates or not response.candidates[0].content.parts:
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
    if not gemini_models_pool or not company_name:
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
{job_text[:40000]}
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
3. For well-known companies (Google, Microsoft, TCS, Infosys etc.), the trust score should be VERY HIGH (90-100).
4. If ownership/CEO details are not easily found but the company is a well-known global brand (e.g., TCS), do NOT penalize the trust score.
4. Include at least 5 factors in trust_breakdown covering: Website, Physical Address, Owner Transparency, Employee Presence, Social Proof, Business Consistency, Registration Proof.
5. The response MUST be valid JSON."""
        
        response = safe_generate(prompt, generation_config={
                "max_output_tokens": 4000, 
                "temperature": 0.3,
                "response_mime_type": "application/json"
            })
        
        if not response:
            raise Exception("All Gemini models exhausted (rate limited)")
        
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
    # Clean markdown links and raw URLs to prevent regex garble
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
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
        skill_items = re.findall(r"[-\u2022*]\s*(.+?)(?:\n|$)", raw_skills)
        if not skill_items:
            skill_items = [s.strip() for s in raw_skills.split(",") if s.strip()]
        details["skills"] = [s.strip()[:80] for s in skill_items[:8]]
    
    return details


def extract_job_details_with_gemini(text):
    """Uses Gemini AI to extract structured job details accurately from raw text.
    Falls back to regex extraction if Gemini is unavailable."""
    regex_details = extract_job_details(text)
    
    if not gemini_models_pool:
        return regex_details
    
    try:
        prompt = f"""You are a data extraction tool. From the following job posting text, extract all structured job details.

Instructions:
- Return ONLY valid JSON, no markdown, no extra text.
- Extract details ONLY if they are clearly mentioned in the text. Do NOT guess or hallucinate.
- If a field is not found or says "not disclosed", DO NOT omit it. Set it to "Not Disclosed" or "Not Specified".
- Ensure values are clean text, NOT markdown links or URLs.
- For skills, extract individual skill items as a list of short strings. If none found, return an empty array [].

Text:
\"\"\"
{text[:40000]}
\"\"\"

Return JSON with exactly these fields:
{{
  "title": "Job Title",
  "location": "Job location",
  "salary": "Salary/CTC/compensation. IMPORTANT: ONLY extract if a currency symbol (₹, $, etc.) or 'LPA/CTC' is explicitly near a number. If it says 'Not Disclosed' or 'Competitive', return 'Not Disclosed'. DO NOT guess or use numbers like '15' or '16' from experience fields as salary.",
  "employment_type": "Full Time / Part Time / Contract / Freelance / Internship / Remote",
  "experience": "Experience required",
  "contact": "Phone, email, or website for contact (or 'Not Specified')",
  "skills": ["skill1", "skill2", ...]
}}"""
        
        response = safe_generate(prompt, generation_config={
                "max_output_tokens": 1000,
                "temperature": 0.1,
                "response_mime_type": "application/json"
            })
        
        if not response:
            print("[WARN] All Gemini models exhausted for job details, using regex fallback")
            return regex_details
        
        response_text = response.text.strip()
        gemini_details = None
        
        try:
            gemini_details = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    gemini_details = json.loads(json_match.group(1).strip())
                except json.JSONDecodeError:
                    pass
            if gemini_details is None:
                first_brace = response_text.find('{')
                last_brace = response_text.rfind('}')
                if first_brace != -1 and last_brace > first_brace:
                    try:
                        gemini_details = json.loads(response_text[first_brace:last_brace + 1])
                    except json.JSONDecodeError:
                        pass
        
        if gemini_details:
            # Merge: prefer Gemini results but keep regex as fallback for missing fields
            merged = {}
            for key in ["title", "location", "salary", "employment_type", "experience", "contact", "skills"]:
                gemini_val = gemini_details.get(key)
                regex_val = regex_details.get(key)
                if gemini_val is not None:
                    if isinstance(gemini_val, list):
                        merged[key] = gemini_val if len(gemini_val) > 0 else (regex_val or [])
                    elif isinstance(gemini_val, str) and gemini_val.strip() and gemini_val.strip().lower() not in ["none", "n/a", "unknown"]:
                        merged[key] = gemini_val.strip()
                    else:
                        merged[key] = regex_val
                else:
                    merged[key] = regex_val
            print(f"[INFO] Gemini job details extracted: {list(merged.keys())}")
            return merged
        else:
            print("[WARN] Gemini job details parsing failed, using regex fallback")
            return regex_details
    except Exception as e:
        print(f"[WARN] Gemini job detail extraction failed: {e}")
        return regex_details


def generate_detailed_summary_with_gemini(text, prediction, confidence, highlights, red_flags):
    """Uses Gemini AI to generate a comprehensive detailed summary and analysis.
    Falls back to template-based summary if Gemini is unavailable."""
    is_fake = prediction == 1
    fallback = get_detailed_analysis(text, prediction, confidence)
    
    if not gemini_models_pool:
        return fallback
    
    try:
        prompt = f"""You are an expert job fraud analyst. Analyze the following job posting and provide a comprehensive report.

Job Posting Text:
\"\"\"
{text[:40000]}
\"\"\"

ML Model Prediction: {"FRAUDULENT" if is_fake else "LEGITIMATE"} (Model Confidence: {confidence * 100:.1f}%)
Top Linguistic Markers: {', '.join(highlights[:5]) if highlights else 'None identified'}

You MUST respond in valid JSON format only. Use this exact structure:
{{
  "summary": "A detailed 3-5 sentence analysis of this job posting. Explain WHY it appears to be {'fraudulent' if is_fake else 'legitimate'}. Reference specific phrases, patterns, or red flags from the actual text. Be specific and insightful, not generic.",
  "red_flags": ["Specific red flag 1 from the text", "Specific red flag 2", ...],
  "company_insight": "A specific insight about the company/employer based on the posting content. Reference actual details from the text.",
  "recommendation": "Clear, actionable recommendation for the job seeker based on this specific posting.",
  "risk_rating": 45
}}

IMPORTANT: 
- The "risk_rating" should be a number from 0 to 100 representing the probability of fraud based on your context-aware analysis. 
- If the company is a well-known brand like TCS and mentions "No security deposit", the risk_rating should be VERY LOW (0-10).

IMPORTANT:
- Be SPECIFIC. Reference actual content from the job posting, not generic statements.
- DO NOT hallucinate salary numbers. If the text says "Not Disclosed", DO NOT invent a number.
- If the company is a high-reputation brand like TCS, be extremely careful about flagging "impersonation" unless there is clear evidence (e.g., a suspicious gmail.com contact or request for money).
- If the posting is fake, identify the exact manipulative tactics used.
- If legitimate, highlight what makes it trustworthy.
- Keep red_flags as an empty array [] if the posting appears legitimate.
- The summary should read like a professional analyst's report."""
        
        response = safe_generate(prompt, generation_config={
                "max_output_tokens": 1500,
                "temperature": 0.3,
                "response_mime_type": "application/json"
            })
        
        if not response:
            print("[WARN] All Gemini models exhausted for summary, using template fallback")
            return fallback
        
        response_text = response.text.strip()
        gemini_analysis = None
        
        try:
            gemini_analysis = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    gemini_analysis = json.loads(json_match.group(1).strip())
                except json.JSONDecodeError:
                    pass
            if gemini_analysis is None:
                first_brace = response_text.find('{')
                last_brace = response_text.rfind('}')
                if first_brace != -1 and last_brace > first_brace:
                    try:
                        gemini_analysis = json.loads(response_text[first_brace:last_brace + 1])
                    except json.JSONDecodeError:
                        pass
        
        if gemini_analysis:
            result = {
                "summary": gemini_analysis.get("summary", fallback["summary"]),
                "red_flags": gemini_analysis.get("red_flags", fallback["red_flags"]),
                "company_insight": gemini_analysis.get("company_insight", fallback["company_insight"]),
                "recommendation": gemini_analysis.get("recommendation", fallback["recommendation"]),
                "risk_rating": gemini_analysis.get("risk_rating", 50 if is_fake else 20),
                "source": "gemini"
            }
            print(f"[INFO] Gemini detailed summary generated successfully")
            return result
        else:
            print("[WARN] Gemini summary parsing failed, using template fallback")
            return fallback
    except Exception as e:
        print(f"[WARN] Gemini detailed summary generation failed: {e}")
        return fallback

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
    
    # Step 3: Pattern Scanning (Context-Aware)
    checks_passed = []
    checks_failed = []
    
    im_check = re.search(r"(?<!no\s)(?<!not\s)(?<!never\s)(?<!don't\s)(?<!doesn't\s)whatsapp|telegram|imo|messenger", low)
    if im_check:
        checks_failed.append("IM platform contact detected")
    else:
        checks_passed.append("No suspicious contact methods")
    
    fee_check = re.search(r"(?<!no\s)(?<!not\s)(?<!never\s)(?<!don't\s)(?<!doesn't\s)deposit|registration fee|processing fee|security fee|refundable", low)
    if fee_check:
        checks_failed.append("Upfront payment request detected")
    else:
        checks_passed.append("No upfront fees requested")
    
    id_check = re.search(r"(?<!no\s)(?<!not\s)(?<!never\s)(?<!don't\s)(?<!doesn't\s)bank account|aadhaar|pan card|cvv|otp", low)
    if id_check:
        checks_failed.append("Sensitive data request detected")
    else:
        checks_passed.append("No early ID/financial data requests")
    
    scam_check = re.search(r"(?<!no\s)(?<!not\s)(?<!never\s)(?<!don't\s)(?<!doesn't\s)daily pay|earn fast|no interview|immediate join|no experience|earn daily|hit like|comment interested|no investment", low)
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
        "detail": f"Prediction: {'FRAUDULENT' if prediction == 1 else 'LEGITIMATE'} (Model Confidence: {round(confidence * 100, 1)}%)",
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
    # Check for high-risk markers (Heuristic Engine)
    highlights = []
    risk_keywords = [
        "whatsapp", "telegram", "registration fee", "security deposit", 
        "processing fee", "bank details", "otp", "password",
        "no interview", "urgent hiring", "lottery", "easy money",
        "send money", "payment", "crypto", "bitcoin", "investment"
    ]
    
    # Context-aware risk detection: Only flag if not preceded by "no", "not", "don't", etc.
    for kw in risk_keywords:
        pattern = r"(?<!no\s)(?<!not\s)(?<!never\s)(?<!don't\s)(?<!doesn't\s)" + re.escape(kw)
        if re.search(pattern, cleaned, re.IGNORECASE):
            highlights.append(kw)
    
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
    # Context-aware red flag detection for the detailed analysis
    if re.search(r"(?<!no\s)(?<!not\s)(?<!never\s)(?<!don't\s)(?<!doesn't\s)whatsapp|telegram|imo|messenger", low_text):
        red_flags.append("Communication via instant messaging apps instead of official channels.")
    if re.search(r"(?<!no\s)(?<!not\s)(?<!never\s)(?<!don't\s)(?<!doesn't\s)deposit|registration fee|processing fee|security fee|refundable", low_text):
        red_flags.append("Request for upfront payment or 'security deposits'.")
    if re.search(r"(?<!no\s)(?<!not\s)(?<!never\s)(?<!don't\s)(?<!doesn't\s)bank account|aadhaar|pan card|cvv|otp", low_text):
        red_flags.append("Requests for sensitive personal or financial identification early in the process.")
    if re.search(r"(?<!no\s)(?<!not\s)(?<!never\s)(?<!don't\s)(?<!doesn't\s)daily pay|earn fast|no interview|immediate join|earn daily|hit like|comment interested|no investment", low_text):
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

    # HEURISTIC OVERRIDE: Context-aware check to prevent false positives on "No security deposit"
    low_text = text.lower()
    override_flags = 0
    if re.search(r"(?<!no\s)(?<!not\s)(?<!never\s)(?<!don't\s)(?<!doesn't\s)whatsapp|telegram|imo|messenger", low_text): override_flags += 1
    if re.search(r"(?<!no\s)(?<!not\s)(?<!never\s)(?<!don't\s)(?<!doesn't\s)deposit|registration fee|processing fee|security fee|refundable|starter kit", low_text): override_flags += 2
    if re.search(r"(?<!no\s)(?<!not\s)(?<!never\s)(?<!don't\s)(?<!doesn't\s)bank account|aadhaar|pan card|cvv|otp", low_text): override_flags += 2
    if re.search(r"(?<!no\s)(?<!not\s)(?<!never\s)(?<!don't\s)(?<!doesn't\s)daily pay|earn fast|no interview|earn daily|hit like|comment interested|no investment", low_text): override_flags += 2

    if override_flags > 0:
        if prediction == 0:
            print(f"[SECURITY] Heuristic Engine overrode ML prediction (Fake). Flags: {override_flags}")
            prediction = 1
        # Boost confidence for heuristic intercepts minimum 90%
        if prediction == 1 and confidence < 0.90:
            confidence = min(0.99, 0.88 + (override_flags * 0.02))

    highlights = get_highlights(text)
    
    # Use Gemini for detailed summary (falls back to template if unavailable)
    analysis = generate_detailed_summary_with_gemini(text, prediction, confidence, highlights, [])
    
    # Extract company name (use Gemini if available, else fallback to regex)
    company_name = extract_company_name_with_gemini(text)
    
    # NEW: Run company verification immediately to calculate a combined Fraud Score
    if company_name:
        company_info = verify_company_with_gemini(company_name, text)
    else:
        company_info = {
            "name": "Unknown",
            "trust_score": 0,
            "verdict": "No company name detected in the job posting. This is a common trait of fraudulent listings.",
            "red_flags": ["No company name provided — legitimate employers always identify themselves."],
            "trust_breakdown": [
                {"factor": "Company Identity", "status": "fail", "detail": "No company name found in posting"}
            ],
            "source": "none"
        }
    
    # --- CALCULATE ADVANCED FRAUD SCORE ---
    # ml_fraud_prob: Base probability of fraud from the ML model
    ml_fraud_prob = probs[1] 
    
    # If heuristics overrode ML, boost the fraud probability
    if override_flags > 0 and prediction == 1:
        ml_fraud_prob = max(ml_fraud_prob, 0.75 + (min(4, override_flags) * 0.05))
    
    ml_component = ml_fraud_prob * 100
    
    # Company trust component (Inverse of trust score)
    trust_score = company_info.get("trust_score") if company_info else None
    if trust_score is not None:
        company_penalty = (100 - trust_score)
    else:
        company_penalty = 50 # Neutral/Unknown
    
    # Red Flag component
    # We differentiate between "detected red flags" (heuristic/AI) and their impact
    analysis_flags = analysis.get("red_flags", [])
    company_flags = company_info.get("red_flags", []) if company_info else []
    total_red_flags = list(set(analysis_flags + company_flags)) # Unique flags
    red_flags_count = len(total_red_flags)
    
    # Penalty based on red flags count (exponentially increasing)
    flag_penalty = min(40, (red_flags_count ** 1.5) * 5) if red_flags_count > 0 else 0
    
    # --- WEIGHTED CALCULATION ---
    # We use a weighted average: 30% ML, 30% Company, 20% Flags, 20% Gemini Opinion
    gemini_rating = analysis.get("risk_rating", 50 if prediction == 1 else 10)
    
    if trust_score and trust_score >= 85:
        # For High Trust companies, we trust Gemini and Company Score more
        fraud_score = (ml_component * 0.15) + (company_penalty * 0.40) + (flag_penalty * 0.15) + (gemini_rating * 0.30)
        
        # SPECIAL OVERRIDE: If trust is extremely high (>90) and red flags are low (<=2)
        if trust_score >= 90 and red_flags_count <= 2:
            prediction = 0
            # If Gemini also thinks it's low risk, drop even further
            if gemini_rating <= 20:
                fraud_score = min(5.0, fraud_score)
            else:
                fraud_score = min(12.0, fraud_score)
    else:
        # Standard weighted calculation
        fraud_score = (ml_component * 0.30) + (company_penalty * 0.25) + (flag_penalty * 0.25) + (gemini_rating * 0.20)
    
    # Apply heuristic "overrides" directly to score if they were found
    # CRITICAL FIX: Only apply these hard-flags if the company trust is low or unknown.
    # Legitimate companies like TCS often mention "No security deposits" which triggers false positives.
    if override_flags > 1 and (trust_score is None or trust_score < 80):
        fraud_score = max(fraud_score, 75.0)
        prediction = 1
    
    # --- FINAL VERDICT CLAMPING ---
    # Ensure consistency between the 'Fake/Real' label and the score
    if prediction == 1: # Flagged as Fake
        if red_flags_count > 2:
            fraud_score = max(70.0, fraud_score) # High risk if > 2 flags
        else:
            fraud_score = max(51.0, fraud_score) # Moderate risk
    else: # Flagged as Real
        if red_flags_count == 0 and trust_score and trust_score > 80:
            fraud_score = min(15.0, fraud_score) # Very safe
        else:
            fraud_score = min(49.0, fraud_score) # Likely safe
    
    fraud_score = min(99.9, max(0.1, fraud_score))

    # Extract job details (use Gemini if available, else fallback to regex)
    job_details = extract_job_details_with_gemini(text)
    
    # Build analysis pipeline
    pipeline = get_analysis_pipeline(
        text, cleaned, prediction, confidence, highlights, analysis.get("red_flags", [])
    )
    
    return {
        "prediction": "Fake" if prediction == 1 else "Real",
        "fraud_score": round(fraud_score, 1),
        "ml_confidence": round(confidence * 100, 1),
        "highlights": highlights,
        "analysis": analysis,
        "extracted_text": text[:500] + "..." if len(text) > 500 else text,
        "company_name": company_name,
        "company_info": company_info,
        "job_details": job_details,
        "pipeline": pipeline,
    }

# --- Endpoints ---
# --- Stats Tracking ---
# Simple in-memory stats for the session (could be persisted to file/DB)
session_stats = {"total": 0, "fake": 0, "real": 0}

@app.route("/stats", methods=["GET"])
def get_stats():
    return jsonify(session_stats)

@app.route("/predict-text", methods=["POST"])
def predict_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data["text"]
    print(f"[INFO] New prediction request: {text[:100]}...")
    
    result = get_prediction_data(text)
    
    if "error" not in result:
        session_stats["total"] += 1
        if result["prediction"] == "Fake":
            session_stats["fake"] += 1
        else:
            session_stats["real"] += 1
    
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "api_keys_count": len(os.getenv("GEMINI_API_KEYS", "").split(",")) if os.getenv("GEMINI_API_KEYS") else 0
    })

@app.route("/predict-image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime = file.content_type or "image/png"

        # 1. NEW STRATEGY: Use Gemini Vision directly for higher accuracy than Tesseract
        print("[INFO] Using Gemini Vision for image analysis")
        vision_prompt = """Extract ALL the text from this job posting image. Preserve the hierarchy if possible. 
        Also, identify the hiring company name explicitly at the very end as 'COMPANY_NAME: [name]'."""
        
        vision_response = safe_generate(vision_prompt, contents=[vision_prompt, img])
        
        if vision_response and vision_response.text:
            raw_text = vision_response.text
            # Extract company name if identified
            extracted_company = None
            if "COMPANY_NAME:" in raw_text:
                extracted_company = raw_text.split("COMPANY_NAME:")[-1].strip()
                raw_text = raw_text.split("COMPANY_NAME:")[0].strip()
            
            result = get_prediction_data(raw_text)
            if extracted_company and (not result.get("company_name") or result["company_name"] == "Unknown"):
                result["company_name"] = extracted_company
        else:
            # Fallback to Tesseract OCR if Gemini Vision fails
            print("[WARN] Gemini Vision failed, falling back to Tesseract OCR")
            extracted_text = pytesseract.image_to_string(img)
            if not extracted_text.strip():
                return jsonify({"error": "Could not extract any text from the image."}), 400
            result = get_prediction_data(extracted_text)
            
        result["image_preview"] = f"data:{mime};base64,{img_b64}"
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Image prediction failed: {e}")
        return jsonify({"error": f"Image analysis failed: {str(e)}"}), 500

@app.route("/predict-url", methods=["POST"])
def predict_url():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "No URL provided"}), 400
    
    url = data["url"].strip()
    if not url.startswith("http"):
        url = "https://" + url
    
    try:
        text = ""
        # 1. Primary Strategy: Use Jina AI Reader API to bypass bot-blocks/JS and extract clean markdown
        try:
            jina_url = "https://r.jina.ai/" + url
            jina_resp = requests.get(jina_url, timeout=20)
            if jina_resp.status_code == 200:
                text = jina_resp.text
                print("[INFO] Successfully extracted page content using Jina AI")
        except Exception as e:
            print(f"[WARN] Jina AI extraction failed, falling back to basic requests: {e}")
            
        # 2. Fallback Strategy: Standard HTML scraping if Jina fails or returned empty
        if not text.strip():
            print("[INFO] Using fallback BeautifulSoup scraper")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
            }
            
            session = requests.Session()
            resp = session.get(url, headers=headers, timeout=15, allow_redirects=True)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.content, "html.parser")
            
            for tag in soup(["script", "style", "noscript", "iframe"]):
                tag.decompose()
            
            if soup.body:
                text = soup.body.get_text(separator="\n")
            else:
                text = soup.get_text(separator="\n")
            
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)
        
        # Truncate very long pages to useful content
        if len(text) > 50000:
            text = text[:50000]
        
        if not text.strip() or len(text.strip()) < 20:
            return jsonify({"error": "Could not extract meaningful text from the webpage. The site may require JavaScript or block automated access."}), 400
            
        result = get_prediction_data(text)
        result["source_url"] = url
        return jsonify(result)
    except requests.exceptions.SSLError:
        return jsonify({"error": "SSL certificate error. The website's security certificate could not be verified."}), 400
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not connect to the website. Please check the URL and try again."}), 400
    except requests.exceptions.Timeout:
        return jsonify({"error": "The website took too long to respond. Please try again later."}), 400
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"Website returned an error: {e.response.status_code}. The page may require login or doesn't exist."}), 400
    except Exception as e:
        print(f"[ERROR] URL scraping failed: {traceback.format_exc()}")
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
    
    if not gemini_models_pool:
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

CHATBOT_SYSTEM_PROMPT = """You are HireGuardAI Career Assistant — an expert AI career counselor embedded in a job fraud detection platform.

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
- IMPORTANT: At the very end of your response, you MUST provide 3 short, relevant follow-up questions the user might ask next. Format them exactly like this at the very bottom:
---SUGGESTIONS---
["question 1", "question 2", "question 3"]

CONTEXT: You have access to the analyzed job posting details below. Use this context to give personalized advice.
"""

def get_fallback_chat_advice(job_context):
    """Provides a safe, hardcoded response if Gemini is unavailable or rate-limited."""
    if not job_context:
        return "I'm sorry, I'm currently experiencing high demand and my AI brain is resting! Please try again in a few minutes. I can help with roadmaps, eligibility, and fraud safety once I'm back online."
    
    company = job_context.get('company_name', 'this company')
    is_fake = job_context.get('prediction') == 'Fake'
    score = job_context.get('fraud_score', 0)
    
    if is_fake:
        return f"⚠️ **Service Notice**: My AI engine is currently over-capacity, but here is my safety assessment for **{company}**:\n\nThis posting has a **high Fraud Score ({score}%)**. I strongly advise you NOT to share Aadhaar/PAN details or pay any 'security deposits'. Legitimate employers like {company} (if this is an impersonation) will never ask for money via WhatsApp or Telegram.\n\n*Please try again in a few minutes for a detailed skill roadmap.*"
    else:
        return f"✅ **Service Notice**: My AI engine is currently over-capacity, but regarding the role at **{company}**:\n\nOur analysis shows a **low Fraud Score ({score}%)**, meaning it appears legitimate. You can proceed with professional caution. I recommend verifying the recruiter's email domain matches the official company website.\n\n*Please try again in a few minutes for interview prep and eligibility tips!*"

@app.route("/chat", methods=["POST"])
def chat():
    if not gemini_models_pool:
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
Prediction: {job_context.get('prediction', 'N/A')} (Fraud Score: {job_context.get('fraud_score', 'N/A')}%, ML Confidence: {job_context.get('ml_confidence', 'N/A')}%)
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
        
        # Try all models in the pool for robustness
        last_error = None
        for model_info in gemini_models_pool:
            try:
                # Switch API key if needed
                global _current_api_key
                if model_info["api_key"] != _current_api_key:
                    genai.configure(api_key=model_info["api_key"])
                    _current_api_key = model_info["api_key"]
                
                # Start chat with history
                chat_obj = model_info["model"].start_chat(history=gemini_history)
                
                # Send current message with system context prepended to first message
                if not gemini_history:
                    full_message = f"{system_context}\n\nUser: {user_message}"
                else:
                    full_message = user_message
                
                response = chat_obj.send_message(full_message)
                
                response_text = response.text
                suggestions = []
                if "---SUGGESTIONS---" in response_text:
                    parts = response_text.split("---SUGGESTIONS---")
                    reply = parts[0].strip()
                    try:
                        suggestions_text = parts[1].strip()
                        if suggestions_text.startswith("```json"): suggestions_text = suggestions_text[7:]
                        elif suggestions_text.startswith("```"): suggestions_text = suggestions_text[3:]
                        if suggestions_text.endswith("```"): suggestions_text = suggestions_text[:-3]
                        suggestions = json.loads(suggestions_text.strip())
                    except Exception as e:
                        print(f"[WARN] Failed to parse follow-up questions: {e}")
                else:
                    reply = response_text
                
                return jsonify({
                    "reply": reply,
                    "suggestions": suggestions,
                    "status": "ok",
                    "model_used": model_info["name"]
                })
                
            except Exception as e:
                last_error = e
                print(f"[WARN] Chat failed with {model_info['key_label']}:{model_info['name']}: {e}")
                continue
        
        # If we reach here, all models failed
        fallback_reply = get_fallback_chat_advice(job_context)
        return jsonify({
            "reply": fallback_reply,
            "suggestions": ["Try again in 2 minutes", "What are general scam signs?", "How to verify a company manually?"],
            "status": "fallback",
            "error": f"API Quota reached. Providing static safety advice."
        })
        
    except Exception as e:
        print(f"[ERROR] Chatbot critical failure: {traceback.format_exc()}")
        return jsonify({"error": f"AI response failed: {str(e)}"}), 500

@app.route("/chat/status", methods=["GET"])
def chat_status():
    """Check if the chatbot is available."""
    return jsonify({
        "available": len(gemini_models_pool) > 0,
        "model": gemini_model_name if gemini_models_pool else None
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
