import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv("backend/.env")

api_keys = os.getenv("GEMINI_API_KEYS", "").split(",")
key = api_keys[0].strip()

genai.configure(api_key=key)

name = "gemini-flash-latest"
try:
    model = genai.GenerativeModel(name)
    response = model.generate_content("test", generation_config={"max_output_tokens": 5})
    print(f"  [OK] {name}: {response.text}")
except Exception as e:
    print(f"  [ERR] {name}: {str(e)}")
