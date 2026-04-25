import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# --- Setup ---
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --- Load ---
print("[INFO] Loading dataset...")
df = pd.read_csv("fake_job_postings.csv")

# --- Clean ---
print("[INFO] Preprocessing text...")
text_cols = ["title", "description", "requirements", "company_profile"]
for col in text_cols:
    df[col] = df[col].fillna("")

df["text"] = df["title"] + " " + df["description"] + " " + df["requirements"] + " " + df["company_profile"]
df["text"] = df["text"].str.lower()
df["text"] = df["text"].apply(lambda t: re.sub(r"[^a-z\s]", "", t))
df["text"] = df["text"].apply(lambda t: " ".join(lemmatizer.lemmatize(w) for w in t.split() if w not in STOP_WORDS))

# --- Vectorize ---
print("[INFO] Vectorizing...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["text"])
y = df["fraudulent"]

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Models ---
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB()
}

results = []

print("[INFO] Training and evaluating models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec
    })
    print(f"   {name} done.")

# --- Plot ---
res_df = pd.DataFrame(results)
res_df.set_index("Model", inplace=True)

ax = res_df.plot(kind="bar", figsize=(10, 6), width=0.8)
plt.title("Model Comparison: Accuracy, Precision & Recall", fontsize=14, fontweight="bold")
plt.ylabel("Score", fontsize=12)
plt.xlabel("Algorithm", fontsize=12)
plt.xticks(rotation=0)
plt.ylim(0, 1.1)
plt.legend(loc="lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
print("[INFO] Comparison plot saved as 'model_comparison.png'")
