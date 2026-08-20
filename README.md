[README .md](https://github.com/user-attachments/files/31271937/README.2.md)
<div align="center">

# 🥗 Nutrition Health Classification

**A machine learning pipeline that classifies food products as Healthy, Moderate, or Unhealthy based on their nutritional profile.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Models](#-models)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [Interactive Interface](#-interactive-interface)
- [Future Work](#-future-work)
- [Team](#-team)
- [License](#-license)

---

## 🔍 Overview

This project predicts how **healthy** a food product is, using only its nutritional values (calories, fat, sugar, protein, etc.). The numeric target `health_score` is converted into a 3-class label `health_class`:

| Score Range | Class |
|---|---|
| `< 40` | Unhealthy |
| `40 – 60` | Moderate |
| `>= 60` | Healthy |

Four classification models were trained and compared, the best one was tuned with `GridSearchCV`, and a live prediction function + interactive `ipywidgets` interface were built on top of it.

---

## 🗂 Dataset

- **Source:** `comprehensive_foods_usda.csv` — USDA comprehensive foods dataset
- **Original size:** 40,000 rows × 24 columns
- **After cleaning:** 40,000 rows × 14 columns *(no exact duplicate rows were found)*

**Numeric (nutritional) features (11):**
`calories`, `fat_g`, `saturated_fat_g`, `sugar_g`, `carbs_g`, `protein_g`, `fiber_g`, `sodium_mg`, `calcium_mg`, `iron_mg`, `cholesterol_mg`

**Categorical features (2):**
`food_category` (240 unique values), `food_type` (9 unique values)

**Target:** `health_score` (range 20–75, mean ≈ 48.9) → converted to `health_class`

**Columns dropped** (identifiers / not predictive): `fdc_id`, `food_name`, `data_type`, `brand_owner`, `brand_name`, `ingredients`, `household_serving`, `serving_size`, `serving_unit`, `vitamin_c_mg`

### Missing values (before imputation)

| Column | Missing Count | Missing % |
|---|---|---|
| `fiber_g` | 6,275 | 15.69% |
| `cholesterol_mg` | 5,842 | 14.61% |
| `calcium_mg` | 5,785 | 14.46% |
| `iron_mg` | 5,750 | 14.38% |
| `saturated_fat_g` | 4,986 | 12.47% |
| `sugar_g` | 4,261 | 10.65% |
| `sodium_mg` | 617 | 1.54% |
| `carbs_g` | 589 | 1.47% |
| `calories` | 462 | 1.16% |
| `fat_g` | 287 | 0.72% |
| `protein_g` | 253 | 0.63% |
| `food_category` | 46 | 0.12% |

All missing values were handled inside the preprocessing pipeline (median imputation for numeric columns, most-frequent for categorical).

### Class distribution (target imbalance)

| Class | Count | % |
|---|---|---|
| Moderate | 27,540 | 68.85% |
| Healthy | 9,076 | 22.69% |
| Unhealthy | 3,384 | 8.46% |

The dataset is noticeably imbalanced toward `Moderate`, which is why `class_weight='balanced'` was used across most models.



---

## ⚙️ Project Workflow

1. **Data Loading** — Load the raw CSV with Pandas.
2. **Data Inspection** — Shape, dtypes, missing values, duplicates.
3. **Data Cleaning**
   - Remove duplicate rows (7300 found).
   - Drop irrelevant / identifier columns.
   - Fix invalid negative values (negative `carbs_g` → clipped to 0).
   - Check for infinite values.
4. **Exploratory Data Analysis (EDA)**
   - Missing value heatmap.
   - Distribution & boxplots of `health_score` and all nutritional features.
   - Correlation matrix between nutritional features and `health_score`.
   - Feature vs. target scatter plots (capped at the 99th percentile).
   - Category-level analysis (`food_category`, `food_type`) vs. health score.
5. **Preprocessing**
   - Outlier handling via capping at the 99th percentile.
   - Target encoding (`health_score` → `health_class`).
   - Stratified train/test split — 32,000 train / 8,000 test (80/20).
   - `ColumnTransformer` pipeline:
     - Numeric: median imputation + `StandardScaler`
     - Categorical: most-frequent imputation + `OneHotEncoder`
6. **Modeling** — Train 4 classifiers (see below).
7. **Evaluation** — Accuracy, weighted F1-score, classification report, confusion matrix for each model.
8. **Hyperparameter Tuning** — `GridSearchCV` (5-fold CV) on the Decision Tree.
9. **Final Model Selection** — Tuned Decision Tree chosen as the production model.
10. **Prediction Function** — `predict_food(...)` for single-record inference.
11. **Interactive Interface** — `ipywidgets`-based UI inside the notebook for live predictions.

---

## 🧠 Models

| Model | Key Settings |
|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight='balanced'` |
| Decision Tree | `max_depth=10`, `class_weight='balanced'` → tuned with `GridSearchCV` |
| Random Forest | `n_estimators=200`, `max_depth=10`, `class_weight='balanced'` |
| K-Nearest Neighbors (KNN) | `n_neighbors=7` |

**Final model: Tuned Decision Tree Classifier** — best combination of accuracy, F1-score, and interpretability. See [Results](#-results) for a note on the very high accuracy.

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| [`pandas`](https://pandas.pydata.org/) | Data loading, cleaning, and manipulation |
| [`numpy`](https://numpy.org/) | Numerical operations, handling inf/NaN values |
| [`matplotlib`](https://matplotlib.org/) | Plotting (histograms, boxplots, bar charts) |
| [`seaborn`](https://seaborn.pydata.org/) | Statistical visualizations (heatmaps, boxplots, countplots) |
| [`scikit-learn`](https://scikit-learn.org/) | Core ML library — pipelines, preprocessing, models, tuning, metrics |
| &nbsp;&nbsp;↳ `train_test_split` | Stratified train/test splitting |
| &nbsp;&nbsp;↳ `ColumnTransformer` | Combining preprocessing for numeric & categorical columns |
| &nbsp;&nbsp;↳ `Pipeline` | Chaining preprocessing + model into one object |
| &nbsp;&nbsp;↳ `SimpleImputer` | Missing value imputation |
| &nbsp;&nbsp;↳ `StandardScaler` | Feature scaling |
| &nbsp;&nbsp;↳ `OneHotEncoder` | Categorical feature encoding |
| &nbsp;&nbsp;↳ `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `KNeighborsClassifier` | Classification models |
| &nbsp;&nbsp;↳ `GridSearchCV` | Hyperparameter tuning |
| &nbsp;&nbsp;↳ `accuracy_score`, `classification_report`, `f1_score` | Evaluation metrics |
| &nbsp;&nbsp;↳ `confusion_matrix`, `ConfusionMatrixDisplay` | Confusion matrix visualization |
| [`ipywidgets`](https://ipywidgets.readthedocs.io/) | Interactive UI components (dropdowns, number inputs, buttons) |
| [`IPython.display`](https://ipython.readthedocs.io/) | Rendering HTML / controlling notebook output |

---

## 📁 Project Structure

```
nutrition-health-classification/
│
├── nutrition_classification.ipynb   # Main notebook (full pipeline)
├── data/
│   └── comprehensive_foods_usda.csv # Raw dataset (not included)
├── visualization/                   # All report figures, organized by category
│   └── ...                          # (multiple subfolders)
├── requirements.txt                 # Project dependencies
├── README.md
└── REPORT.md                        # Detailed technical report
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/nutrition-health-classification.git
cd nutrition-health-classification
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Place `comprehensive_foods_usda.csv` inside a `data/` folder, and update the loading path in the notebook:
```python
df = pd.read_csv("data/comprehensive_foods_usda.csv")
```

### 5. Run the notebook
```bash
jupyter notebook nutrition_classification.ipynb
```
Run all cells in order — the last section launches the interactive prediction interface.

**`requirements.txt`**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
ipywidgets
jupyter
```

---

## 📊 Results

All four models were evaluated on the same 8,000-row held-out test set.

| Model | Test Accuracy | Weighted F1-Score |
|---|---|---|
| Logistic Regression | 82.23% | 0.833 |
| KNN (k=7) | 91.11% | 0.911 |
| Random Forest | 93.56% | 0.938 |
| **Decision Tree** | **99.89%** | **0.999** |

### Hyperparameter tuning (Decision Tree, `GridSearchCV`, 5-fold CV)

- **Best parameters:** `max_depth=10`, `min_samples_leaf=2`, `min_samples_split=2`
- **Best CV F1-score (weighted):** 0.9996
- **Final tuned test accuracy:** 99.89% — identical to the untuned baseline, confirming the model was already near its ceiling.

**Final classification report (tuned Decision Tree, test set):**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Healthy | 1.00 | 1.00 | 1.00 | 1,815 |
| Moderate | 1.00 | 1.00 | 1.00 | 5,508 |
| Unhealthy | 1.00 | 1.00 | 1.00 | 677 |

The Decision Tree's near-perfect score reflects how cleanly the `health_class` boundaries separate the nutritional feature space once outliers are capped and the data is properly preprocessed — the class thresholds on `health_score` align closely with learnable splits in the underlying nutrition values.

---

## 🖥 Interactive Interface

The final section of the notebook includes a built-in UI (via `ipywidgets`) that lets a user:
- Select any of the four trained models.
- Enter nutritional values for a food item.
- Choose a food category and type from dropdowns auto-populated from the dataset.
- Get an instant prediction (`Healthy` / `Moderate` / `Unhealthy`) with styled visual feedback.

---

## 🔮 Future Work

- Try gradient-boosting models (XGBoost, LightGBM, CatBoost) for additional comparison.
- Engineer additional features (nutrient ratios, composite indices).
- Deploy as a standalone web app (Streamlit / Flask / FastAPI) instead of an in-notebook widget UI.
- Apply broader cross-validation across all models, not just the Decision Tree.

---

## 👥 Team

This project was built collaboratively by:

- **Romysaa Mohamed Qotb** 
- **Mazen Hussein AL-Borkan** 
- **Nour EL-Din Abdel Khalek Khalil**
- **Bavly Nashaat Nageh**
- **Mohmoud Abdel Hamid Abdel Rahman** 


---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
