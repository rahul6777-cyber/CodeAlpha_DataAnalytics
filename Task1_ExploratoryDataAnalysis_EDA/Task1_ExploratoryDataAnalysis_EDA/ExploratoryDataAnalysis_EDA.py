# ============================================================
#  TASK 2: Exploratory Data Analysis (EDA) — Titanic Dataset
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Load Data ────────────────────────────────────────────────
url = (
    "https://raw.githubusercontent.com/datasciencedojo/"
    "datasets/master/titanic.csv"
)
df = pd.read_csv(url)

print("=" * 60)
print("  TASK 2: Exploratory Data Analysis — Titanic")
print("=" * 60)

# ── 1. Meaningful Questions ──────────────────────────────────
print("\n❓ Meaningful Questions Before Analysis:")
questions = [
    "1. What was the overall survival rate?",
    "2. Did gender affect survival chances?",
    "3. Did passenger class influence survival?",
    "4. How did age vary between survivors and non-survivors?",
    "5. Are there missing values that could bias analysis?",
]
for q in questions:
    print(f"   {q}")

# ── 2. Data Structure ────────────────────────────────────────
print("\n🔍 Data Structure:")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
print(f"   {'Column':<15} {'Dtype':<12} {'Unique':<8} {'Nulls'}")
print(f"   {'-'*45}")
for col in df.columns:
    print(f"   {col:<15} {str(df[col].dtype):<12} {df[col].nunique():<8} {df[col].isnull().sum()}")

# ── 3. Descriptive Statistics ────────────────────────────────
print("\n📊 Descriptive Statistics:")
print(df.describe().round(2).to_string())

# ── 4. Trends, Patterns & Anomalies ─────────────────────────
print("\n📈 Trends & Patterns:")
surv_rate   = df["Survived"].mean() * 100
female_surv = df[df["Sex"] == "female"]["Survived"].mean() * 100
male_surv   = df[df["Sex"] == "male"]["Survived"].mean()   * 100
print(f"   • Overall survival rate : {surv_rate:.1f}%")
print(f"   • Female survival rate  : {female_surv:.1f}%")
print(f"   • Male survival rate    : {male_surv:.1f}%")

print("\n   Survival by Passenger Class:")
for pclass, rate in (df.groupby("Pclass")["Survived"].mean() * 100).items():
    print(f"   • Class {pclass}: {rate:.1f}%")

print("\n   Age Stats:")
print(f"   • Mean age : {df['Age'].mean():.1f}")
print(f"   • Median age : {df['Age'].median():.1f}")
print(f"   • Age range : {df['Age'].min():.0f} – {df['Age'].max():.0f}")

# ── 5. Hypotheses & Assumptions ─────────────────────────────
print("\n🧪 Hypotheses Tested:")
children = df[df["Age"] < 16]["Survived"].mean() * 100
adults   = df[df["Age"] >= 16]["Survived"].mean() * 100
print(f"   H1: Women survived more than men → CONFIRMED ({female_surv:.1f}% vs {male_surv:.1f}%)")

pclass_surv = df.groupby("Pclass")["Survived"].mean() * 100
result = "CONFIRMED" if pclass_surv[1] > pclass_surv[3] else "NOT CONFIRMED"
print(f"   H2: 1st class survived more than 3rd → {result} ({pclass_surv[1]:.1f}% vs {pclass_surv[3]:.1f}%)")

result = "CONFIRMED" if children > adults else "NOT CONFIRMED"
print(f"   H3: Children (<16) survived more than adults → {result} ({children:.1f}% vs {adults:.1f}%)")

# ── 6. Data Issues ───────────────────────────────────────────
print("\n⚠️  Data Issues Detected:")
missing = df.isnull().sum()
for col, count in missing[missing > 0].items():
    pct = count / len(df) * 100
    action = "Consider dropping" if pct > 50 else "Impute with median/mode"
    print(f"   • '{col}': {count} nulls ({pct:.1f}%) → {action}")

print("\n" + "=" * 60)
print("  EDA COMPLETE")
print("=" * 60)
