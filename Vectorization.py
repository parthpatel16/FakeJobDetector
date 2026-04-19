# ============================================================
# Fake Job Postings — TF-IDF Vectorization & Train-Test Split
# ============================================================
# This script picks up after Preprocessing.py:
#   1. Loads the dataset and applies the same text cleaning
#   2. Converts cleaned text to TF-IDF features
#   3. Splits into 80% train / 20% test
#   4. Demonstrates class_weight='balanced' for imbalanced data
#
# Libraries : pandas, nltk, scikit-learn
# Dataset   : fake_job_postings.csv
# ============================================================

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------
# 0. NLTK setup
# ----------------------------------------------------------
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# ----------------------------------------------------------
# 1. Load and preprocess (same steps as Preprocessing.py)
# ----------------------------------------------------------
df = pd.read_csv("fake_job_postings.csv")

text_cols = ["title", "description", "requirements", "company_profile"]
for col in text_cols:
    df[col] = df[col].fillna("")

# Combine → lowercase → remove special chars → remove stopwords
df["text"] = (
    df["title"] + " " + df["description"] + " " +
    df["requirements"] + " " + df["company_profile"]
)
df["text"] = df["text"].str.lower()
df["text"] = df["text"].apply(lambda t: re.sub(r"[^a-z\s]", "", t))
df["text"] = df["text"].apply(
    lambda t: " ".join(w for w in t.split() if w not in STOP_WORDS)
)

print("✅ Text preprocessing complete.\n")

# ----------------------------------------------------------
# 2. Define features (X) and target (y)
# ----------------------------------------------------------
X = df["text"]                # cleaned text column
y = df["fraudulent"]          # 0 = real, 1 = fake

# ----------------------------------------------------------
# 3. TF-IDF Vectorization
# ----------------------------------------------------------
# max_features=5000 keeps the top 5000 terms by frequency,
# which is a good balance between information and speed.
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

print("✅ TF-IDF vectorization complete.")
print(f"   Vocabulary size (max_features) : {X_tfidf.shape[1]}")
print(f"   Sparse matrix shape            : {X_tfidf.shape}\n")

# ----------------------------------------------------------
# 4. Train-Test Split (80-20)
# ----------------------------------------------------------
# stratify=y ensures both sets keep the same fraud ratio.
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("✅ Data split into train and test sets.")
print(f"   X_train shape : {X_train.shape}")
print(f"   X_test  shape : {X_test.shape}")
print(f"   y_train counts: Real={sum(y_train == 0)}, Fake={sum(y_train == 1)}")
print(f"   y_test  counts: Real={sum(y_test == 0)}, Fake={sum(y_test == 1)}\n")

# ----------------------------------------------------------
# 5. Class imbalance — class_weight='balanced'
# ----------------------------------------------------------
# The dataset is highly imbalanced (~95% real vs ~5% fake).
# When training a classifier, use class_weight='balanced'
# so the model penalises mistakes on the minority class more.
#
# Example usage with common classifiers:
#
#   from sklearn.linear_model import LogisticRegression
#   model = LogisticRegression(class_weight='balanced')
#
#   from sklearn.ensemble import RandomForestClassifier
#   model = RandomForestClassifier(class_weight='balanced')
#
#   from sklearn.svm import LinearSVC
#   model = LinearSVC(class_weight='balanced')
#
# The 'balanced' option automatically adjusts weights
# inversely proportional to class frequencies:
#   weight_i = n_samples / (n_classes * n_samples_i)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.array([0, 1])
weights = compute_class_weight("balanced", classes=classes, y=y_train)

print("✅ Class weights computed (for 'balanced' mode):")
print(f"   Class 0 (Real) weight : {weights[0]:.4f}")
print(f"   Class 1 (Fake) weight : {weights[1]:.4f}")
print("   → The model will pay ~{:.0f}× more attention to fake postings.\n".format(
    weights[1] / weights[0]
))

print("=" * 60)
print("READY FOR MODEL TRAINING")
print("=" * 60)
