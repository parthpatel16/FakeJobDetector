import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv("backend/.env")

api_keys = os.getenv("GEMINI_API_KEYS", "").split(",")
key = api_keys[0].strip()

print(f"Testing Key: {key[:10]}...")
genai.configure(api_key=key)

test_names = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
]

for name in test_names:
    try:
        model = genai.GenerativeModel(name)
        response = model.generate_content("test", generation_config={"max_output_tokens": 5})
        print(f"  [OK] {name}: {response.text}")
    except Exception as e:
        print(f"  [ERR] {name}: {str(e)}")
