from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# --- Load Model and Vectorizer ---
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("Γ£à Model and Vectorizer loaded successfully.")
except Exception as e:
    print(f"Γ¥î Error loading models: {e}")

# --- Text Preprocessing (Must match training) ---
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

def preprocess_text(text):
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove non-alphabet characters
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in STOP_WORDS])
    return text

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Get JSON data
        data = request.get_json()
        if not data or "description" not in data:
            return jsonify({"error": "Please provide 'description' in JSON"}), 400
        
        job_description = data["description"]
        
        # 2. Preprocess
        cleaned_text = preprocess_text(job_description)
        
        # 3. Vectorize
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # 4. Predict
        prediction = model.predict(vectorized_text)[0]
        # Get probability (confidence)
        # model.predict_proba returns [[prob_0, prob_1]]
        probabilities = model.predict_proba(vectorized_text)[0]
        confidence = float(max(probabilities))
        
        # 5. Return result
        result = {
            "prediction": "Fake" if prediction == 1 else "Real",
            "confidence_score": round(confidence, 4)
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the app
    app.run(debug=True, port=5000)
