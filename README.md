# 🚢 Titanic Dataset — Complete Data Cleaning & Preprocessing Pipeline

> A production-grade machine learning data preprocessing project demonstrating every critical step from raw data to model-ready features, with clear visualizations and reproducible results.

---

## 📋 Table of Contents

1. [Project Objectives](#project-objectives)
2. [Dataset Overview](#dataset-overview)
3. [Project Structure](#project-structure)
4. [Preprocessing Workflow](#preprocessing-workflow)
5. [Techniques Used](#techniques-used)
6. [Key Observations](#key-observations)
7. [Final Outcomes](#final-outcomes)
8. [Setup & Reproduction](#setup--reproduction)
9. [Interview Q&A](#interview-qa)

---

## 🎯 Project Objectives

- Perform end-to-end data cleaning and preprocessing on the Titanic dataset
- Handle missing values using statistically appropriate imputation techniques
- Convert categorical variables to numerical format via Label Encoding and One-Hot Encoding
- Apply feature scaling (Min-Max Normalization & Z-score Standardization)
- Detect and treat outliers using the IQR method
- Produce clear, publication-quality visualizations at each stage
- Document every decision with explanation and code

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| Source | Seaborn built-in (`sns.load_dataset("titanic")`) |
| Rows | 891 |
| Columns | 15 (raw), 10 (final) |
| Target | `survived` (binary: 0 / 1) |

### Features Used

| Feature | Type | Missing | Treatment |
|---|---|---|---|
| survived | int (target) | 0 | — |
| pclass | int (ordinal) | 0 | — |
| sex | string | 0 | Label Encoding |
| age | float | 177 (20%) | Median Imputation |
| sibsp | int | 0 | IQR scaling |
| parch | int | 0 | IQR scaling |
| fare | float | 0 | IQR Capping |
| embarked | string | 2 | Mode Imputation + OHE |

---

## 📁 Project Structure

```
titanic-preprocessing/
│
├── data/
│   ├── titanic.csv                  # Raw dataset
│   └── titanic_preprocessed.csv     # Final clean dataset
│
├── src/
│   └── titanic_preprocessing.py     # Full pipeline script
│
├── plots/
│   ├── 01_eda_overview.png          # EDA: survival, missing, distributions
│   ├── 02_missing_values.png        # Missing values before/after
│   ├── 02b_age_imputation.png       # Age distribution before/after imputation
│   ├── 03_encoding.png              # Encoding visualizations
│   ├── 04_scaling.png               # Scaling comparisons (3 rows × 4 features)
│   ├── 05_outliers.png              # Boxplots before/after IQR capping
│   └── 06_correlation_heatmap.png   # Final feature correlation matrix
│
└── README.md
```

---

## 🔄 Preprocessing Workflow

### Step 1 — Exploratory Data Analysis (EDA)

**Goal:** Understand the dataset's structure, distributions, and quality before making any changes.

**Actions:**
- Loaded 891 rows × 15 columns via `seaborn`
- Inspected data types (int, float, str, category, bool)
- Computed summary statistics (`describe(include="all")`)
- Identified missing values: `age` (177), `embarked` (2), `deck` (688, dropped)

**Visual:** `01_eda_overview.png`

---

### Step 2 — Handling Missing Data

**Goal:** Fill gaps without introducing bias or distorting the underlying distribution.

| Column | Method | Reason |
|---|---|---|
| `age` | **Median** imputation | Skewed distribution; median is robust to extremes |
| `embarked` | **Mode** imputation | Nominal category; most frequent class is safest default |
| `deck` | **Dropped** | 77% missing — imputation would be misleading |

**Code snippet:**
```python
df["age"].fillna(df["age"].median(), inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
```

**Visual:** `02_missing_values.png`, `02b_age_imputation.png`

---

### Step 3 — Categorical Encoding

**Goal:** Convert string/categorical features to numbers that ML algorithms can process.

| Column | Method | Justification |
|---|---|---|
| `sex` | **Label Encoding** | Binary (female=0, male=1) — no ordinality issue |
| `embarked` | **One-Hot Encoding** | Nominal 3-class — OHE avoids false ordinal assumptions |

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["sex_encoded"] = le.fit_transform(df["sex"])

ohe = pd.get_dummies(df["embarked"], prefix="emb")
df  = pd.concat([df, ohe], axis=1)
```

**Visual:** `03_encoding.png`

---

### Step 4 — Feature Scaling

**Goal:** Bring all numeric features to a comparable scale so distance-based and gradient-based algorithms aren't dominated by large-magnitude features.

| Method | Formula | When to Use |
|---|---|---|
| **Min-Max Normalization** | `(x − min) / (max − min)` | Bounded output [0,1]; useful for neural networks, k-NN |
| **Z-score Standardization** | `(x − μ) / σ` | Preserves outlier information; good for linear models, SVM |

Both were applied to: `age`, `fare`, `sibsp`, `parch`.

**Visual:** `04_scaling.png`

---

### Step 5 — Outlier Detection & Treatment

**Goal:** Identify extreme values that could distort model training.

**Method: IQR (Interquartile Range) Capping**

```
Lower bound = Q1 − 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
Values outside bounds are CLIPPED (Winsorization)
```

| Feature | Outliers Found | Bounds |
|---|---|---|
| `age` | 11 | [−6.69, 64.81] → capped at valid range |
| `fare` | 116 | [−26.72, 65.63] |

> **Capping** (Winsorization) was chosen over deletion to preserve row count and avoid data loss. Fare had 116 outliers — removing them would have significantly reduced the dataset.

**Visual:** `05_outliers.png`

---

### Step 6 — Final Correlation Analysis

Heatmap of the fully preprocessed dataset reveals:
- `pclass` and `fare` are strongly negatively correlated (−0.55)
- `sex_encoded` (male=1) is negatively correlated with `survived` (−0.54)
- `pclass` is negatively correlated with `survived` (−0.34)

**Visual:** `06_correlation_heatmap.png`

---

## 🛠 Techniques Used

| Category | Technique | Library |
|---|---|---|
| Missing values | Median / Mode imputation | pandas |
| Encoding | Label Encoding | sklearn.preprocessing |
| Encoding | One-Hot Encoding | pandas.get_dummies |
| Scaling | Min-Max Normalization | sklearn.preprocessing.MinMaxScaler |
| Scaling | Z-score Standardization | sklearn.preprocessing.StandardScaler |
| Outlier detection | IQR method | numpy / pandas |
| Outlier treatment | Capping (Winsorization) | pandas.clip |
| Visualization | Distribution plots, heatmap, boxplots | matplotlib, seaborn |

---

## 🔍 Key Observations

1. **Age had 20% missing values** — the median (28 years) was a better imputer than mean (29.7) because the distribution had a mild right skew.
2. **`deck` was 77% missing** and was dropped entirely — imputation at this rate would manufacture data, not clean it.
3. **`fare` had 116 outliers** (13% of data) — several passengers paid extremely high fares (max £512 vs. IQR cap at £65.6). Capping preserves the rows while bounding the extreme influence.
4. **Label Encoding was appropriate for `sex`** (binary feature) but would introduce false ordinality if applied to `embarked` (C < Q < S would be meaningless).
5. **Min-Max scaling changed the shape** of the fare distribution visually but preserved relative relationships; Z-score made all distributions comparable in terms of standard deviations.

---

## 📈 Final Outcomes

| Metric | Value |
|---|---|
| Raw shape | (891, 15) |
| Final shape | (891, 10) |
| Missing values | 0 |
| Categorical columns | 0 |
| Numeric features scaled | 4 |
| Outliers treated | 127 (age + fare) |
| Dropped columns | 5 (deck, embark_town, class, who, alive, alone) |

The final dataset (`titanic_preprocessed.csv`) is fully numeric, zero-null, scaled, and ready for any ML algorithm.

---

## ⚙️ Setup & Reproduction

### Requirements
```
python >= 3.9
pandas, numpy, matplotlib, seaborn, scikit-learn
```

### Install
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run
```bash
python src/titanic_preprocessing.py
```

All plots will be saved to `plots/` and the clean dataset to `data/titanic_preprocessed.csv`.

---

## 💬 Interview Q&A

### Q1. What are the types of missing data?

There are three fundamental types:

| Type | Description | Example |
|---|---|---|
| **MCAR** (Missing Completely At Random) | Missingness is unrelated to any variable | Random sensor failure |
| **MAR** (Missing At Random) | Missingness is related to observed variables but not the missing value itself | Age missing more for males (sex is observed) |
| **MNAR** (Missing Not At Random) | Missingness is related to the unobserved value itself | High-income people less likely to report income |

**In this project:** `age` is likely MAR; `deck` is MNAR (lower-class passengers less likely to have cabin records).

---

### Q2. How do you handle categorical variables?

| Scenario | Technique | Reason |
|---|---|---|
| Binary (yes/no, male/female) | **Label Encoding** | No false ordering introduced |
| Nominal with 3+ classes | **One-Hot Encoding** | Avoids integer ordering artifacts |
| Ordinal (low < medium < high) | **Ordinal Encoding** | Preserves meaningful rank |
| High cardinality (100+ classes) | **Target Encoding / Embedding** | OHE becomes too wide |

---

### Q3. Normalization vs Standardization — when to use which?

| | Normalization (Min-Max) | Standardization (Z-score) |
|---|---|---|
| **Formula** | (x−min)/(max−min) | (x−μ)/σ |
| **Output range** | [0, 1] | Unbounded, mean=0 std=1 |
| **Preserves outliers** | Yes (squishes them) | Partially |
| **Use when** | Neural networks, k-NN, image pixels | Linear/logistic regression, SVM, PCA |
| **Sensitive to outliers** | Yes | Less so |

**Rule of thumb:** Use standardization by default; use normalization when the algorithm explicitly requires [0,1] input.

---

### Q4. What are the main outlier detection methods?

1. **IQR Method** — Flagged as outlier if < Q1−1.5×IQR or > Q3+1.5×IQR. Non-parametric, robust.
2. **Z-score** — Outlier if |z| > 3. Assumes normal distribution.
3. **Boxplots** — Visual detection of IQR-based outliers.
4. **Isolation Forest** — Tree-based ML method; good for high-dimensional data.
5. **DBSCAN** — Density-based; identifies points in low-density regions as outliers.
6. **Local Outlier Factor (LOF)** — Compares local density of a point to its neighbours.

---

### Q5. Why is preprocessing important in machine learning?

1. **Garbage in, garbage out** — Models cannot fix bad input; they amplify noise.
2. **Algorithm requirements** — Many algorithms (linear regression, SVM, k-NN) assume scaled, numeric features.
3. **Missing values cause errors** — Most sklearn estimators raise exceptions on NaN inputs.
4. **Outliers distort optimization** — Gradient descent and distance metrics are highly sensitive.
5. **Feature scale matters** — Without scaling, a feature measured in thousands can dominate one measured in decimals, regardless of predictive power.
6. **Encoding is mandatory** — Neural networks, trees, and linear models cannot operate on strings.

---

### Q6. One-Hot Encoding vs Label Encoding

| | One-Hot Encoding | Label Encoding |
|---|---|---|
| **Output** | N binary columns | 1 integer column |
| **Best for** | Nominal categories | Binary / ordinal features |
| **Risk** | Curse of dimensionality for high cardinality | False ordering (A=0 < B=1 < C=2 is meaningless if nominal) |
| **Memory** | Higher (sparse matrix) | Lower |
| **Algorithms** | Linear models, neural nets | Tree-based (handles integers natively) |

**Dummy variable trap:** When using OHE, drop one column (`drop_first=True`) to avoid perfect multicollinearity with linear models.

---

### Q7. How do you handle class imbalance?

| Method | Description |
|---|---|
| **SMOTE** | Synthetic Minority Oversampling — creates synthetic minority samples |
| **Random Oversampling** | Duplicate minority samples randomly |
| **Random Undersampling** | Remove majority samples |
| **Class weights** | Pass `class_weight="balanced"` to sklearn estimators |
| **Threshold tuning** | Adjust classification threshold (e.g., 0.3 instead of 0.5) |
| **Ensemble methods** | BalancedRandomForest, EasyEnsemble |

The Titanic dataset has ~38% survival rate, which is a mild imbalance. For severe imbalance (>1:10), SMOTE or class weighting is critical.

---

### Q8. How does preprocessing impact model accuracy?

| Preprocessing Step | Affected Models | Impact |
|---|---|---|
| **Missing value imputation** | All | Enables training; wrong imputation adds noise |
| **Feature scaling** | k-NN, SVM, Linear/Logistic Regression, Neural Nets | Can improve accuracy by 5–30% |
| **Encoding** | All | Required for non-tree models |
| **Outlier treatment** | Linear models, k-NN | Prevents extreme coefficients / distances |
| **Feature engineering** | All | Often the biggest accuracy gain |

> **Decision trees and random forests** are largely scale-invariant and handle mixed types better than most models — but they still benefit from outlier treatment and clean data.

---

*Project by: Data Preprocessing Pipeline | Dataset: Titanic (Seaborn)*
