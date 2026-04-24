import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv("backend/.env")

api_keys = os.getenv("GEMINI_API_KEYS", "").split(",")
key = api_keys[0].strip()

print(f"Listing models for Key: {key[:10]}...")
genai.configure(api_key=key)

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name}")
except Exception as e:
    print(f"Error: {e}")
