R"""
=============================================================
  Titanic Dataset — Complete Data Cleaning & Preprocessing
=============================================================
Author : Data Preprocessing Project
Dataset: Titanic (seaborn built-in)
"""

# ── Imports ───────────────────────────────────────────────
import warnings, os
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot  as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# ── Paths ─────────────────────────────────────────────────
BASE   = os.getcwd() # Changed from os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS  = os.path.join(BASE, "plots")
DATA   = os.path.join(BASE, "data")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

PALETTE = ["#2E86AB","#E84855","#3BB273","#F7B731","#9B5DE5","#F15BB5"]
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.titleweight": "bold"})

# ═══════════════════════════════════════════════════════════
# 1. LOAD & EDA
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 1 — LOAD DATASET & EXPLORATORY DATA ANALYSIS")
print("="*60)

df_raw = sns.load_dataset("titanic")
df     = df_raw.copy()

print(f"\n▸ Shape          : {df.shape}")
print(f"▸ Columns ({len(df.columns)})  : {df.columns.tolist()}")
print("\n▸ Data Types:\n", df.dtypes)
print("\n▸ Summary Statistics:\n", df.describe(include="all").T)
print("\n▸ Missing Values:\n", df.isnull().sum()[df.isnull().sum() > 0])

# ── Plot 1 : EDA overview ─────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Step 1 — Exploratory Data Analysis", fontsize=16, y=1.01)

# Survival distribution
axes[0,0].bar(["Did Not Survive","Survived"],
              df["survived"].value_counts().values,
              color=PALETTE[:2], edgecolor="white", linewidth=1.5)
axes[0,0].set_title("Survival Distribution"); axes[0,0].set_ylabel("Count")
for bar, v in zip(axes[0,0].patches, df["survived"].value_counts().values):
    axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                   str(v), ha="center", fontweight="bold")

# Missing values heatmap
miss = df.isnull().sum().sort_values(ascending=False)
miss = miss[miss > 0]
axes[0,1].barh(miss.index, miss.values, color=PALETTE[1])
axes[0,1].set_title("Missing Values per Column")
axes[0,1].set_xlabel("Count")

# Age distribution
axes[0,2].hist(df["age"].dropna(), bins=30, color=PALETTE[0],
               edgecolor="white", linewidth=0.8)
axes[0,2].axvline(df["age"].mean(), color=PALETTE[1], linestyle="--",
                  label=f"Mean={df['age'].mean():.1f}")
axes[0,2].axvline(df["age"].median(), color=PALETTE[2], linestyle="--",
                  label=f"Median={df['age'].median():.1f}")
axes[0,2].set_title("Age Distribution (before imputation)")
axes[0,2].legend()

# Fare distribution
axes[1,0].hist(df["fare"], bins=40, color=PALETTE[3], edgecolor="white")
axes[1,0].set_title("Fare Distribution"); axes[1,0].set_xlabel("Fare")

# Pclass vs Survived
ct = df.groupby("pclass")["survived"].mean().reset_index()
axes[1,1].bar(ct["pclass"].astype(str), ct["survived"],
              color=PALETTE[:3], edgecolor="white")
axes[1,1].set_title("Survival Rate by Pclass")
axes[1,1].set_ylabel("Survival Rate")

# Sex vs Survived
ct2 = df.groupby("sex")["survived"].mean().reset_index()
axes[1,2].bar(ct2["sex"], ct2["survived"], color=PALETTE[4:6], edgecolor="white")
axes[1,2].set_title("Survival Rate by Sex")
axes[1,2].set_ylabel("Survival Rate")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "01_eda_overview.png"))
plt.close()
print("\n✔  Plot saved: 01_eda_overview.png")


# ═══════════════════════════════════════════════════════════
# 2. HANDLE MISSING DATA
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 2 — HANDLING MISSING DATA")
print("="*60)

# Keep only ML-relevant columns
COLS = ["survived","pclass","sex","age","sibsp","parch","fare","embarked"]
df   = df[COLS].copy()

before_miss = df.isnull().sum().copy()

# Age → median imputation (right-skewed)
age_median = df["age"].median()
df["age"].fillna(age_median, inplace=True)
print(f"▸ 'age'      — imputed {before_miss['age']} NaN with MEDIAN ({age_median:.1f})")

# Embarked → mode imputation (categorical)
emb_mode = df["embarked"].mode()[0]
df["embarked"].fillna(emb_mode, inplace=True)
print(f"▸ 'embarked' — imputed {before_miss['embarked']} NaN with MODE ('{emb_mode}')")

print("\n▸ Missing values after imputation:\n", df.isnull().sum())

# ── Plot 2 : Before vs After imputation ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Step 2 — Missing Data Imputation", fontsize=14)

axes[0].barh(before_miss.index, before_miss.values, color=PALETTE[1])
axes[0].set_title("Missing Values — BEFORE"); axes[0].set_xlabel("Count")

after_miss = df.isnull().sum()

axes[1].barh(
    after_miss.index,
    after_miss.values + 0.001,
    color=PALETTE[2]
)

axes[1].set_xlim(0, max(before_miss.values)+10)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "02_missing_values.png"))
plt.close()
print("✔  Plot saved: 02_missing_values.png")


# ── Plot 2b : Age distribution before vs after ────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Step 2b — Age Distribution Before vs After Imputation", fontsize=13)

axes[0].hist(df_raw["age"].dropna(), bins=30, color=PALETTE[0], edgecolor="white")
axes[0].set_title(f"Before  (n={df_raw['age'].notna().sum()})"); axes[0].set_xlabel("Age")

axes[1].hist(df["age"], bins=30, color=PALETTE[2], edgecolor="white")
axes[1].axvline(age_median, color=PALETTE[1], linestyle="--", label=f"Median={age_median}")
axes[1].set_title(f"After  (n={len(df)})")
axes[1].set_xlabel("Age")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "02b_age_imputation.png"))
plt.close()
print("✔  Plot saved: 02b_age_imputation.png")


# ═══════════════════════════════════════════════════════════
# 3. ENCODING CATEGORICAL VARIABLES
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 3 — CATEGORICAL ENCODING")
print("="*60)

df_enc = df.copy()

# Label Encoding → binary 'sex'
le = LabelEncoder()
df_enc["sex_encoded"] = le.fit_transform(df_enc["sex"])
print(f"▸ Label Encoding  'sex'      : {dict(zip(le.classes_, le.transform(le.classes_)))}")

# One-Hot Encoding → nominal 'embarked'
ohe = pd.get_dummies(df_enc["embarked"], prefix="emb")
df_enc = pd.concat([df_enc, ohe], axis=1)
print(f"▸ One-Hot Encoding 'embarked': {list(ohe.columns)}")

# Drop originals
df_enc.drop(columns=["sex","embarked"], inplace=True)
print("\n▸ Encoded dataframe shape:", df_enc.shape)
print(df_enc.head(3))

# ── Plot 3 : Encoding comparison ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Step 3 — Categorical Variable Encoding", fontsize=14)

vc = df["sex"].value_counts()
axes[0].bar(vc.index, vc.values, color=PALETTE[4:6], edgecolor="white")
axes[0].set_title("Original 'sex' (string)"); axes[0].set_ylabel("Count")

vc2 = df_enc["sex_encoded"].value_counts().sort_index()
axes[1].bar(["0 = female","1 = male"], vc2.values, color=PALETTE[4:6], edgecolor="white")
axes[1].set_title("Label-Encoded 'sex'")

ohe_sums = ohe.sum()
axes[2].bar(ohe_sums.index, ohe_sums.values,
            color=PALETTE[:3], edgecolor="white")
axes[2].set_title("One-Hot Encoded 'embarked'")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "03_encoding.png"))
plt.close()
print("✔  Plot saved: 03_encoding.png")


# ═══════════════════════════════════════════════════════════
# 4. FEATURE SCALING
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 4 — FEATURE SCALING")
print("="*60)

NUM_FEATS = ["age","fare","sibsp","parch"]

# Min-Max Normalization
mm = MinMaxScaler()
df_minmax = df_enc.copy()
df_minmax[NUM_FEATS] = mm.fit_transform(df_enc[NUM_FEATS])
print("▸ Min-Max Normalization applied → values in [0,1]")
print(df_minmax[NUM_FEATS].describe().round(3))

# Z-score Standardization
ss = StandardScaler()
df_std = df_enc.copy()
df_std[NUM_FEATS] = ss.fit_transform(df_enc[NUM_FEATS])
print("\n▸ Z-score Standardization applied → mean≈0, std≈1")
print(df_std[NUM_FEATS].describe().round(3))

# ── Plot 4 : Scaling comparison ───────────────────────────
fig, axes = plt.subplots(3, len(NUM_FEATS), figsize=(16, 11))
fig.suptitle("Step 4 — Feature Scaling: Original vs MinMax vs Z-score",
             fontsize=14)

for i, feat in enumerate(NUM_FEATS):
    kw = dict(bins=30, edgecolor="white", alpha=0.85)
    axes[0,i].hist(df_enc[feat],   color=PALETTE[0], **kw)
    axes[0,i].set_title(f"{feat} — Original")

    axes[1,i].hist(df_minmax[feat], color=PALETTE[2], **kw)
    axes[1,i].set_title(f"{feat} — MinMax [0,1]")

    axes[2,i].hist(df_std[feat],   color=PALETTE[3], **kw)
    axes[2,i].set_title(f"{feat} — Z-score")

    for r in range(3):
        axes[r,i].set_xlabel(feat)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "04_scaling.png"))
plt.close()
print("\n✔  Plot saved: 04_scaling.png")


# ═══════════════════════════════════════════════════════════
# 5. OUTLIER DETECTION & TREATMENT
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 5 — OUTLIER DETECTION & TREATMENT (IQR)")
print("="*60)

OUTLIER_FEATS = ["age","fare"]

def iqr_bounds(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR    = Q3 - Q1
    return Q1 - 1.5*IQR, Q3 + 1.5*IQR

df_clean = df_enc.copy()
outlier_report = {}

for feat in OUTLIER_FEATS:
    lo, hi = iqr_bounds(df_clean[feat])
    n_out  = ((df_clean[feat] < lo) | (df_clean[feat] > hi)).sum()
    outlier_report[feat] = {"lower": lo, "upper": hi, "n_outliers": n_out}
    print(f"▸ {feat:<6} — IQR bounds [{lo:.2f}, {hi:.2f}]  |  outliers: {n_out}")
    df_clean[feat] = df_clean[feat].clip(lower=lo, upper=hi)

print(f"\n▸ Rows before / after outlier capping: {len(df_enc)} / {len(df_clean)}")

# ── Plot 5 : Boxplots before & after ─────────────────────
fig, axes = plt.subplots(2, len(OUTLIER_FEATS), figsize=(12, 10))
fig.suptitle("Step 5 — Outlier Detection & Treatment (IQR Capping)", fontsize=14)

for i, feat in enumerate(OUTLIER_FEATS):

    bx0 = axes[0,i].boxplot(
        [df_enc[feat].dropna().values],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2)
    )
    bx0["boxes"][0].set_facecolor(PALETTE[1])

    axes[0,i].set_title(
        f"{feat} — BEFORE\n(outliers={outlier_report[feat]['n_outliers']})"
    )
    axes[0,i].set_ylabel(feat)

    bx1 = axes[1,i].boxplot(
        [df_clean[feat].dropna().values],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2)
    )
    bx1["boxes"][0].set_facecolor(PALETTE[2])

    axes[1,i].set_title(f"{feat} — AFTER (IQR Capped)")
    axes[1,i].set_ylabel(feat)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "05_outliers.png"))
plt.close()
print("✔  Plot saved: 05_outliers.png")


# ═══════════════════════════════════════════════════════════
# 6. CORRELATION HEATMAP (FINAL)
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 6 — FINAL CORRELATION HEATMAP")
print("="*60)

# Scale final clean df
df_final = df_clean.copy()
df_final[NUM_FEATS] = mm.fit_transform(df_final[NUM_FEATS])

fig, ax = plt.subplots(figsize=(11, 8))
carr = df_final.corr()
mask = np.triu(np.ones_like(carr, dtype=bool))
sns.heatmap(carr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
            cmap="coolwarm", center=0, ax=ax, square=True,
            cbar_kws={"shrink": 0.8})
ax.set_title("Final Correlation Heatmap — Preprocessed Titanic Data", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "06_correlation_heatmap.png"))
plt.close()
print("✔  Plot saved: 06_correlation_heatmap.png")


# ── Save final clean CSV ──────────────────────────────────
df_final.to_csv(os.path.join(DATA, "titanic_preprocessed.csv"), index=False)
print(f"\n✔  Final dataset saved: data/titanic_preprocessed.csv  {df_final.shape}")

print("\n" + "="*60)
print("  PREPROCESSING COMPLETE — All plots saved to /plots")
print("="*60)
