# ============================================================
# Fake Job Postings — Logistic Regression Model Training
# ============================================================
# This script trains a RandomForest classifier on the
# TF-IDF features produced by Vectorization.py, evaluates it
# on the held-out test set, and visualises the confusion matrix.
#
# Libraries : pandas, nltk, scikit-learn, matplotlib, numpy
# Dataset   : fake_job_postings.csv
# ============================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# =============================================================
# PART A — Reproduce preprocessing & vectorization pipeline
# =============================================================
# (Same steps as Preprocessing.py + Vectorization.py so this
#  script is fully self-contained and runnable on its own.)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# --- Load ---
df = pd.read_csv("fake_job_postings.csv")

# --- Clean text ---
text_cols = ["title", "description", "requirements", "company_profile"]
for col in text_cols:
    df[col] = df[col].fillna("")

df["text"] = (
    df["title"] + " " + df["description"] + " " +
    df["requirements"] + " " + df["company_profile"]
)
df["text"] = df["text"].str.lower()
df["text"] = df["text"].apply(lambda t: re.sub(r"[^a-z\s]", "", t))
df["text"] = df["text"].apply(
    lambda t: " ".join(lemmatizer.lemmatize(w) for w in t.split() if w not in STOP_WORDS)
)

# --- TF-IDF ---
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df["text"])
y = df["fraudulent"]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

print("[DATA] Data ready.")
print(f"   X_train: {X_train.shape}  |  X_test: {X_test.shape}\n")

# =============================================================
# PART B — Model Training
# =============================================================

# ----------------------------------------------------------
# 1. Train Logistic Regression with class_weight='balanced'
# ----------------------------------------------------------
# 'balanced' automatically adjusts weights inversely
# proportional to class frequencies, giving ~20× more
# importance to the minority "Fake" class (class 1).

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",   # handle class imbalance
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("[Trained] RandomForest model trained.\n")

# ----------------------------------------------------------
# 2. Predict on the test set
# ----------------------------------------------------------
y_pred = model.predict(X_test)

# =============================================================
# PART C — Evaluation Metrics
# =============================================================

# ----------------------------------------------------------
# 3. Individual metrics
# ----------------------------------------------------------
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)       # for class 1
recall    = recall_score(y_test, y_pred)           # for class 1
f1        = f1_score(y_test, y_pred)               # for class 1

print("=" * 60)
print("MODEL EVALUATION METRICS")
print("=" * 60)
print(f"   Accuracy  : {accuracy:.4f}   ({accuracy*100:.2f}%)")
print(f"   Precision : {precision:.4f}   ({precision*100:.2f}%)")
print(f"   Recall    : {recall:.4f}   ({recall*100:.2f}%)")
print(f"   F1 Score  : {f1:.4f}   ({f1*100:.2f}%)")
print()

# ----------------------------------------------------------
# 4. Classification Report (per-class breakdown)
# ----------------------------------------------------------
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(
    y_test, y_pred,
    target_names=["Real (0)", "Fake (1)"]
))

# ----------------------------------------------------------
# 5. Confusion Matrix (numeric)
# ----------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print(f"   True Negatives  (Real -> Real) : {cm[0][0]}")
print(f"   False Positives (Real -> Fake) : {cm[0][1]}")
print(f"   False Negatives (Fake -> Real) : {cm[1][0]}")
print(f"   True Positives  (Fake -> Fake) : {cm[1][1]}")
print()

# =============================================================
# PART D — Confusion Matrix Plot
# =============================================================

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Real (0)", "Fake (1)"],
)
disp.plot(cmap="Blues", ax=ax, values_format="d")

ax.set_title("Confusion Matrix — RandomForest\n(class_weight='balanced')",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("[INFO] Confusion matrix saved as 'confusion_matrix.png'")
plt.show(block=False)
plt.pause(2)
plt.close()

# =============================================================
# PART E — Why Recall Matters in Fake Job Detection
# =============================================================
print("=" * 60)
print("WHY RECALL IS IMPORTANT HERE")
print("=" * 60)
print("""
In fake job detection, a False Negative means a FRAUDULENT
posting was classified as REAL. That is dangerous because:

  • Job seekers may share personal data (resume, ID, bank
    details) with a scammer, leading to identity theft.
  • Missing even one fake posting can cause real harm.

RECALL = TP / (TP + FN)
  -> It measures: "Of all actual fake postings, how many
    did we correctly catch?"

High recall ensures we catch as many fake postings as
possible, even if it means flagging a few real ones
(lower precision) — because the cost of missing a scam
is far greater than the cost of double-checking a
legitimate post.

In summary:
  Precision -> "How many flagged posts are truly fake?"
  Recall    -> "How many fake posts did we actually catch?"
  -> For safety, RECALL is the priority metric.
""")

# =============================================================
# PART F — Top Words for FAKE vs REAL Jobs
# =============================================================
# The logistic regression model learns one coefficient per
# TF-IDF feature.  A large POSITIVE coefficient means the word
# pushes the prediction toward class 1 (FAKE), while a large
# NEGATIVE coefficient pushes toward class 0 (REAL).

# ----------------------------------------------------------
# 1. Get feature names from the TfidfVectorizer
# ----------------------------------------------------------
feature_names = np.array(tfidf.get_feature_names_out())
print("[INFO] Feature names extracted from TfidfVectorizer.")
print(f"   Total features: {len(feature_names)}\n")

# ----------------------------------------------------------
# 2. Extract coefficients from the trained model
# ----------------------------------------------------------
# RandomForest uses feature_importances_ instead of coef_
importances = model.feature_importances_
print("[INFO] Feature importances extracted from RandomForest model.")
print(f"   Importance array shape: {importances.shape}\n")

# ----------------------------------------------------------
# 3. Sort features by coefficient value
# ----------------------------------------------------------
sorted_indices = np.argsort(importances)
top_indices = sorted_indices[-20:][::-1]   # 20 most important features

# ----------------------------------------------------------
# 4. Display Top 20 words indicating FAKE jobs
# ----------------------------------------------------------
print("=" * 60)
print("TOP 20 MOST IMPORTANT WORDS IN THE MODEL")
print("=" * 60)
print(f"  {'Rank':<6} {'Word':<25} {'Importance':>12}")
print("-" * 60)
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank:<6} {feature_names[idx]:<25} {importances[idx]:>12.4f}")
print()

# ----------------------------------------------------------
# 5. Plot & Save Top 20 Feature Importances
# ----------------------------------------------------------
plt.figure(figsize=(10, 8))
plt.barh(feature_names[top_indices][::-1], importances[top_indices][::-1], color='skyblue')
plt.xlabel('Importance Score')
plt.title('Top 20 Most Important Words in Fraud Detection')
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("[INFO] Feature importance plot saved as 'feature_importance.png'\n")
plt.close()

# =============================================================
# PART G — Save & Load Model with Pickle
# =============================================================
import pickle

# ----------------------------------------------------------
# 1. Save the trained Logistic Regression model
# ----------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("[INFO] RandomForest model saved as 'model.pkl'")

# ----------------------------------------------------------
# 2. Save the fitted TfidfVectorizer
# ----------------------------------------------------------
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
print("[INFO] TfidfVectorizer saved as 'vectorizer.pkl'\n")

# ----------------------------------------------------------
# 3. Load them back (demonstration)
# ----------------------------------------------------------
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    loaded_vectorizer = pickle.load(f)

print("[INFO] Model and Vectorizer loaded back successfully.")
print(f"   Loaded model type       : {type(loaded_model).__name__}")
print(f"   Loaded vectorizer type  : {type(loaded_vectorizer).__name__}")
print(f"   Vectorizer vocab size   : {len(loaded_vectorizer.vocabulary_)}")

# Quick sanity check — predict on first test sample
sample_pred = loaded_model.predict(X_test[0])
print(f"   Sample prediction (1st test row): {sample_pred[0]}  "
      f"({'Fake' if sample_pred[0] == 1 else 'Real'})")
print()
