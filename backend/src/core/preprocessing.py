# ============================================================
# Fake Job Postings — Text Preprocessing for ML
# ============================================================
# This script cleans and prepares the raw job-posting text
# so it is ready for machine-learning tasks such as
# classification, clustering, or NLP feature extraction.
#
# Pipeline:
#   1. Select relevant text columns
#   2. Fill missing values
#   3. Combine columns into a single 'text' field
#   4. Lowercase the text
#   5. Remove punctuation & special characters
#   6. Remove English stopwords (NLTK)
#   7. Preview cleaned output
#
# Libraries : pandas, nltk, re
# Dataset   : fake_job_postings.csv
# ============================================================

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# ----------------------------------------------------------
# 0. One-time NLTK download (silent if already present)
# ----------------------------------------------------------
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# ----------------------------------------------------------
# 1. Load the dataset
# ----------------------------------------------------------
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "fake_job_postings.csv"))
print("✅ Dataset loaded.")
print(f"   Rows: {df.shape[0]}  |  Columns: {df.shape[1]}\n")

# ----------------------------------------------------------
# 2. Select important text columns
# ----------------------------------------------------------
# These four columns carry the most textual information
# about each job posting and are useful for ML models.
text_cols = ["title", "description", "requirements", "company_profile"]

# ----------------------------------------------------------
# 3. Fill missing values with empty strings
# ----------------------------------------------------------
# Many rows have NaN in one or more text fields.
# Replacing them with "" prevents 'NaN' literals from
# leaking into the combined text.
for col in text_cols:
    df[col] = df[col].fillna("")

print("✅ Missing values filled with empty strings.")

# ----------------------------------------------------------
# 4. Combine all text columns into a single 'text' column
# ----------------------------------------------------------
# A single unified column is easier for vectorizers
# (TF-IDF, CountVectorizer, etc.) to process.
df["text"] = (
    df["title"]           + " " +
    df["description"]     + " " +
    df["requirements"]    + " " +
    df["company_profile"]
)

print("✅ Text columns combined into 'text'.")

# ----------------------------------------------------------
# 5. Convert text to lowercase
# ----------------------------------------------------------
# Ensures "Manager" and "manager" are treated as the
# same token during feature extraction.
df["text"] = df["text"].str.lower()

print("✅ Text converted to lowercase.")

# ----------------------------------------------------------
# 6. Remove punctuation and special characters
# ----------------------------------------------------------
# Keep only lowercase letters and whitespace.
# Digits, HTML entities, URLs, etc. are all stripped out.
df["text"] = df["text"].apply(
    lambda t: re.sub(r"[^a-z\s]", "", t)
)

print("✅ Punctuation and special characters removed.")

# ----------------------------------------------------------
# 7. Remove stopwords
# ----------------------------------------------------------
# Common English words ("the", "is", "in", …) add noise
# without carrying useful meaning for classification.
df["text"] = df["text"].apply(
    lambda t: " ".join(
        word for word in t.split() if word not in STOP_WORDS
    )
)

print("✅ Stopwords removed.\n")

# ----------------------------------------------------------
# 8. Show a sample of cleaned text
# ----------------------------------------------------------
print("=" * 60)
print("SAMPLE OF CLEANED TEXT (first 3 rows)")
print("=" * 60)
for i in range(3):
    # Show first 300 characters of each sample
    preview = df["text"].iloc[i][:300]
    print(f"\n[Row {i}]\n{preview} …\n")

# ----------------------------------------------------------
# 9. Final summary
# ----------------------------------------------------------
print("=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"   Total rows       : {len(df)}")
print(f"   Target column    : 'fraudulent'  — "
      f"{df['fraudulent'].value_counts().to_dict()}")
print(f"   Text column      : 'text'")
print(f"   Avg word count   : "
      f"{df['text'].str.split().str.len().mean():.0f} words/row")
print()

# ----------------------------------------------------------
# 10. Save preprocessed data (optional — uncomment to use)
# ----------------------------------------------------------
# df.to_csv("preprocessed_job_postings.csv", index=False)
# print("💾 Saved to 'preprocessed_job_postings.csv'")
