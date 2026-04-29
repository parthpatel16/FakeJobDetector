import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_api_keys():
    # Path to the .env file
    env_path = r"c:\Users\PARTH PATEL\OneDrive\Desktop\DAV Project\backend\.env"
    
    # Load environment variables
    load_dotenv(env_path, override=True)
    
    _raw_keys = os.getenv("GEMINI_API_KEYS", "")
    _single_key = os.getenv("GEMINI_API_KEY", "")
    
    api_keys = []
    if _raw_keys:
        api_keys = [k.strip() for k in _raw_keys.split(",") if k.strip()]
    if not api_keys and _single_key:
        api_keys = [_single_key]
        
    if not api_keys:
        print("No API keys found in .env file.")
        return

    print(f"Testing {len(api_keys)} API keys...")
    print("-" * 50)
    
    results = []
    for i, key in enumerate(api_keys):
        key_label = f"Key {i+1} ({key[:8]}...{key[-4:]})"
        print(f"Testing {key_label}...", end=" ", flush=True)
        
        try:
            genai.configure(api_key=key)
            # Try to list models as a simple key validation
            models = genai.list_models()
            model_list = [m.name for m in models]
            
            if model_list:
                print(f"[WORKING] - Found {len(model_list)} models")
                results.append((key_label, f"Working ({len(model_list)} models)"))
            else:
                print("[FAILED] (No models found)")
                results.append((key_label, "Failed (No models found)"))
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                print("[RATE LIMITED]")
                results.append((key_label, "Rate Limited"))
            elif "403" in error_msg or "API_KEY_INVALID" in error_msg:
                print("[INVALID KEY]")
                results.append((key_label, "Invalid Key"))
            else:
                print(f"[ERROR]: {error_msg[:50]}...")
                results.append((key_label, f"Error: {error_msg[:50]}"))

    print("-" * 50)
    print("Summary:")
    for label, status in results:
        print(f"{label}: {status}")

if __name__ == "__main__":
    test_api_keys()
