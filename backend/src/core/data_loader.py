# ============================================================
# Fake Job Postings - Exploratory Data Analysis (EDA)
# ============================================================
# Libraries: pandas, matplotlib
# Dataset : fake_job_postings.csv
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. Load the dataset
# ----------------------------------------------------------
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "fake_job_postings.csv"))
print("✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ----------------------------------------------------------
# 2. Show first 5 rows
# ----------------------------------------------------------
print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())
print()

# ----------------------------------------------------------
# 3. Show column names
# ----------------------------------------------------------
print("=" * 60)
print("COLUMN NAMES")
print("=" * 60)
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")
print()

# ----------------------------------------------------------
# 4. Check missing values
# ----------------------------------------------------------
print("=" * 60)
print("MISSING VALUES (per column)")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
})
print(missing_df)
print(f"\nTotal missing values: {missing.sum()}")
print()

# ----------------------------------------------------------
# 5. Distribution of target column 'fraudulent'
# ----------------------------------------------------------
print("=" * 60)
print("TARGET COLUMN DISTRIBUTION — 'fraudulent'")
print("=" * 60)
print(df["fraudulent"].value_counts())
print()
print(df["fraudulent"].value_counts(normalize=True).round(4) * 100)
print()

# --- Bar chart ---
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["fraudulent"].value_counts()
labels = ["Real (0)", "Fake (1)"]
colors = ["#2ecc71", "#e74c3c"]

bars = ax.bar(labels, counts.values, color=colors, edgecolor="black", width=0.5)

# Add count labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 150,
            f"{int(height)}", ha="center", va="bottom", fontweight="bold", fontsize=12)

ax.set_title("Distribution of Fraudulent Job Postings", fontsize=14, fontweight="bold")
ax.set_xlabel("Class", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_ylim(0, counts.max() + 1500)
plt.tight_layout()
plt.savefig("fraudulent_distribution.png", dpi=150)
plt.show()

print("📊 Chart saved as 'fraudulent_distribution.png'")
