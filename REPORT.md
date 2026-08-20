[REPORT .md](https://github.com/user-attachments/files/31260773/REPORT.2.md)
# 📑 Technical Report: Nutrition Health Classification

## 1. Introduction

This report documents the full development process of a machine learning system that classifies food products into three health categories — `Healthy`, `Moderate`, and `Unhealthy` — based on their nutritional profile. The dataset used is the USDA comprehensive foods dataset, and the entire pipeline was implemented in Python within a Jupyter Notebook.

This is a **multi-class classification** problem, solved end-to-end: from raw data cleaning through model selection, tuning, and deployment as an interactive in-notebook tool.

---

## 2. Dataset

- **Source:** `comprehensive_foods_usda.csv`
- **Original shape:** 40,000 rows × 24 columns
- **Shape after cleaning:** 40,000 rows × 14 columns
- **Duplicate rows found:** 0

**Columns dropped** as not useful for modeling:
`fdc_id`, `food_name`, `data_type`, `brand_owner`, `brand_name`, `ingredients`, `household_serving`, `serving_size`, `serving_unit`, `vitamin_c_mg`

**Numeric (nutritional) features used (11):**
`calories`, `fat_g`, `saturated_fat_g`, `sugar_g`, `carbs_g`, `protein_g`, `fiber_g`, `sodium_mg`, `calcium_mg`, `iron_mg`, `cholesterol_mg`

**Categorical features (2):**
`food_category` (240 unique values), `food_type` (9 unique values)

**Target variable:** `health_score`, numeric, range 20–75, mean ≈ 48.9, std ≈ 9.85 — later converted to the categorical `health_class`.

---

## 3. Data Cleaning

1. **Duplicates** — Checked with `duplicated().sum()`; **0 duplicate rows** were found, so no rows were removed on this basis.
2. **Fully-null columns** — Checked for columns with 100% missing values; none were found.
3. **Irrelevant columns** — Identifier and free-text columns (brand, ingredients, serving metadata, `vitamin_c_mg`) were dropped as they carry no direct predictive signal or were mostly missing.
4. **Invalid negative values** — Negative values were found in `carbs_g` (physically impossible for a nutrient amount) and were clipped to `0`.
5. **Infinite values** — All numeric columns were checked for `inf` values using `np.isinf()`; none were found.
6. **Missing values** — A full missing-value summary was generated:

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

   These were left for the modeling pipeline to handle via imputation (median for numeric, most-frequent for categorical), rather than dropping rows and losing data.

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Missing values

![Missing values heatmap](assets/eda/missing_values_heatmap.png)

Each white streak represents missing data in that column. Confirms the pattern already quantified in the missing-value table above — `fiber_g`, `cholesterol_mg`, `calcium_mg`, `iron_mg`, and `saturated_fat_g` are the most affected columns.

### 4.2 Target variable (`health_score`)

<table>
<tr>
<td><img src="assets/eda/health_score_distribution.png" width="420"/></td>
<td><img src="assets/eda/health_score_boxplot.png" width="420"/></td>
</tr>
</table>

The score distribution is concentrated between 40 and 60, with a strong peak around 50 (13,914 rows at exactly 50 — this single value alone makes up ~35% of the dataset).

### 4.3 Distribution of nutritional features (capped at 99th percentile)

<table>
<tr>
<td><img src="assets/distributions/calories.png" width="270"/></td>
<td><img src="assets/distributions/fat_g.png" width="270"/></td>
<td><img src="assets/distributions/saturated_fat_g.png" width="270"/></td>
</tr>
<tr>
<td><img src="assets/distributions/sugar_g.png" width="270"/></td>
<td><img src="assets/distributions/carbs_g.png" width="270"/></td>
<td><img src="assets/distributions/protein_g.png" width="270"/></td>
</tr>
<tr>
<td><img src="assets/distributions/fiber_g.png" width="270"/></td>
<td><img src="assets/distributions/sodium_mg.png" width="270"/></td>
<td><img src="assets/distributions/calcium_mg.png" width="270"/></td>
</tr>
<tr>
<td><img src="assets/distributions/iron_mg.png" width="270"/></td>
<td><img src="assets/distributions/cholesterol_mg.png" width="270"/></td>
<td></td>
</tr>
</table>

Most nutritional features are heavily right-skewed, which is expected for real-world food composition data (most products are low in a given nutrient, with a long tail of nutrient-dense items).

<details>
<summary><b>Boxplots for all 11 nutritional features (click to expand)</b></summary>

<table>
<tr>
<td><img src="assets/boxplots/calories.png" width="270"/></td>
<td><img src="assets/boxplots/fat_g.png" width="270"/></td>
<td><img src="assets/boxplots/saturated_fat_g.png" width="270"/></td>
</tr>
<tr>
<td><img src="assets/boxplots/sugar_g.png" width="270"/></td>
<td><img src="assets/boxplots/carbs_g.png" width="270"/></td>
<td><img src="assets/boxplots/protein_g.png" width="270"/></td>
</tr>
<tr>
<td><img src="assets/boxplots/fiber_g.png" width="270"/></td>
<td><img src="assets/boxplots/sodium_mg.png" width="270"/></td>
<td><img src="assets/boxplots/calcium_mg.png" width="270"/></td>
</tr>
<tr>
<td><img src="assets/boxplots/iron_mg.png" width="270"/></td>
<td><img src="assets/boxplots/cholesterol_mg.png" width="270"/></td>
<td></td>
</tr>
</table>

These confirm heavy outlier presence in nearly every feature — the justification for percentile-based capping in preprocessing (Section 5).

</details>

### 4.4 Correlation with `health_score`

<table>
<tr>
<td><img src="assets/eda/correlation_matrix.png" width="420"/></td>
<td><img src="assets/eda/correlation_barchart.png" width="420"/></td>
</tr>
</table>

Notable correlations with `health_score`: `saturated_fat_g` (−0.39), `fat_g` (−0.24), `sugar_g` (−0.24), `protein_g` (+0.29), `fiber_g` (+0.25), `calories` (−0.13). Also worth noting: `sugar_g` and `carbs_g` are strongly correlated with each other (0.82), and `fat_g`/`saturated_fat_g` similarly (0.69) — expected multicollinearity between related nutrients.

<details>
<summary><b>Scatter plots: each feature vs. health_score (click to expand)</b></summary>

<table>
<tr>
<td><img src="assets/scatter/calories_vs_health_score.png" width="270"/></td>
<td><img src="assets/scatter/fat_g_vs_health_score.png" width="270"/></td>
<td><img src="assets/scatter/saturated_fat_g_vs_health_score.png" width="270"/></td>
</tr>
<tr>
<td><img src="assets/scatter/sugar_g_vs_health_score.png" width="270"/></td>
<td><img src="assets/scatter/carbs_g_vs_health_score.png" width="270"/></td>
<td><img src="assets/scatter/protein_g_vs_health_score.png" width="270"/></td>
</tr>
<tr>
<td><img src="assets/scatter/fiber_g_vs_health_score.png" width="270"/></td>
<td><img src="assets/scatter/sodium_mg_vs_health_score.png" width="270"/></td>
<td><img src="assets/scatter/calcium_mg_vs_health_score.png" width="270"/></td>
</tr>
<tr>
<td><img src="assets/scatter/iron_mg_vs_health_score.png" width="270"/></td>
<td><img src="assets/scatter/cholesterol_mg_vs_health_score.png" width="270"/></td>
<td></td>
</tr>
</table>

</details>

### 4.5 Categorical analysis

- **240** unique food categories, **9** unique food types.
- Top categories by frequency: *Fruit & Vegetable Juice, Nectars & Fruit Drinks* (1,937), *Cheese* (1,303), *Candy* (1,185), *Breads & Buns* (1,011), *Chocolate* (994).
- Top food types by frequency: *Other* (15,521), *Snacks & Sweets* (4,971), *Dairy* (4,756), *Fruits* (4,566), *Meat & Poultry* (4,211).

<table>
<tr>
<td><img src="assets/categories/top15_food_categories.png" width="420"/></td>
<td><img src="assets/categories/health_score_by_food_category.png" width="420"/></td>
</tr>
<tr>
<td><img src="assets/categories/food_type_distribution.png" width="420"/></td>
<td><img src="assets/categories/health_score_by_food_type.png" width="420"/></td>
</tr>
</table>

### Key Findings
- Features such as sugar and saturated fat show a negative relationship with `health_score`, while features such as fiber and protein tend to correlate positively — consistent with nutritional intuition.
- `health_score` varies noticeably across food categories and food types, confirming both carry useful predictive signal.
- Nearly every nutritional column contains extreme outliers, justifying percentile-based capping before modeling.

---

## 5. Preprocessing

1. **Outlier handling:** Values were **capped** at the 99th percentile for each nutritional feature (rather than removed), to preserve dataset size.
2. **Target encoding:**
   ```python
   def health_category(score):
       if score < 40:
           return 'Unhealthy'
       elif score < 60:
           return 'Moderate'
       else:
           return 'Healthy'
   ```
   Resulting class distribution:

   | Class | Count | % |
   |---|---|---|
   | Moderate | 27,540 | 68.85% |
   | Healthy | 9,076 | 22.69% |
   | Unhealthy | 3,384 | 8.46% |

   The dataset is clearly imbalanced toward `Moderate`, motivating the use of `class_weight='balanced'` in most models.

3. **Train/test split:** 80/20 split with `stratify=y` → 32,000 training rows / 8,000 test rows, preserving class proportions in both sets.
4. **Preprocessing pipeline** via `ColumnTransformer`:
   - **Numeric features:** median imputation → `StandardScaler`
   - **Categorical features:** most-frequent imputation → `OneHotEncoder`

---

## 6. Modeling

Four classification models were trained, each wrapped in a `Pipeline` combining the preprocessor and the classifier:

| # | Model | Key Configuration |
|---|---|---|
| 1 | **Logistic Regression** | `max_iter=1000`, `class_weight='balanced'` |
| 2 | **Decision Tree** | `max_depth=10`, `class_weight='balanced'` |
| 3 | **Random Forest** | `n_estimators=200`, `max_depth=10`, `class_weight='balanced'` |
| 4 | **KNN** | `n_neighbors=7` |

---

## 7. Model Evaluation

All models were evaluated on the same held-out test set of 8,000 rows.

### Logistic Regression
- Test accuracy: **82.23%** | Train accuracy: 82.77%
- Weighted F1-score: 0.833
- Per-class F1: Healthy 0.83, Moderate 0.86, Unhealthy 0.62 — the minority `Unhealthy` class is noticeably weaker (low precision, 0.46), a direct effect of class imbalance even with balanced weighting.

<img src="assets/models/confusion_matrix_logistic_regression.png" width="420"/>

### Decision Tree
- Test accuracy: **99.89%** | Train accuracy: 100%
- Weighted F1-score: 0.999
- Precision/Recall/F1 = 1.00 across all three classes.

<img src="assets/models/confusion_matrix_decision_tree.png" width="420"/>

**Feature importance** (top 15 features driving the Decision Tree's predictions):

<img src="assets/models/feature_importance_decision_tree.png" width="500"/>

### Random Forest
- Test accuracy: **93.56%** | Train accuracy: 94.17%
- Weighted F1-score: 0.938
- Per-class F1: Healthy 0.95, Moderate 0.95, Unhealthy 0.81 — strong recall (1.00) but lower precision (0.69) on `Unhealthy`.

<img src="assets/models/confusion_matrix_random_forest.png" width="420"/>

### KNN (k=7)
- Test accuracy: **91.11%** | Train accuracy: 93.31%
- Weighted F1-score: 0.911
- Most balanced per-class performance of the non-tree models: Healthy 0.87, Moderate 0.94, Unhealthy 0.82.

<img src="assets/models/confusion_matrix_knn.png" width="420"/>

### Consolidated Comparison

| Model | Test Accuracy | Weighted F1-Score |
|---|---|---|
| Logistic Regression | 82.23% | 0.8327 |
| KNN | 91.11% | 0.9108 |
| Random Forest | 93.56% | 0.9383 |
| **Decision Tree** | **99.89%** | **0.9989** |

<img src="assets/models/model_comparison_barchart.png" width="500"/>

---

## 8. Hyperparameter Tuning

The **Decision Tree** was selected for tuning as the strongest baseline model. `GridSearchCV` was used with 5-fold cross-validation, optimizing for weighted F1-score:

```python
param_grid = {
    'classifier__max_depth': [3, 5, 7, 10, 15, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 5]
}
```

**Results:**
- **Best parameters:** `max_depth=10`, `min_samples_leaf=2`, `min_samples_split=2`
- **Best cross-validated F1-score (weighted):** 0.9996
- **Test accuracy after tuning:** 99.89% — essentially unchanged from the untuned Decision Tree, indicating the default `max_depth=10` configuration was already close to optimal for this data.

---

## 9. Final Model

The **tuned Decision Tree Classifier** was selected as the final production model:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Healthy | 1.00 | 1.00 | 1.00 | 1,815 |
| Moderate | 1.00 | 1.00 | 1.00 | 5,508 |
| Unhealthy | 1.00 | 1.00 | 1.00 | 677 |

**Overall test accuracy: 99.89%**

This high score reflects how well-separated the `health_class` boundaries are within the nutritional feature space after preprocessing (outlier capping, imputation, scaling/encoding) — the Decision Tree is able to recover very clean, near-exact splits between the three classes.

A reusable inference function, `predict_food(...)`, accepts a product's nutritional values along with `food_category` and `food_type`, and returns its predicted class.

---

## 10. Interactive Interface

The final section of the notebook implements an interactive UI using `ipywidgets`, including:
- **Model selector:** dropdown to choose among the trained models (Logistic Regression / Decision Tree / Random Forest / KNN).
- **Nutritional inputs:** numeric fields (`FloatText`) for each nutritional value.
- **Category selectors:** dropdowns auto-populated from the unique values in the dataset (`food_category`, `food_type`), defaulting to `Unknown`.
- **Prediction button:** runs the selected model on the entered values and displays the result inline with styled HTML/CSS feedback.

---

## 11. Tools & Libraries

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn` (pipelines, preprocessing, models, metrics, `GridSearchCV`) |
| Interactive UI | `ipywidgets`, `IPython.display` |
| Development environment | Jupyter Notebook |

---

## 12. Conclusion

This project delivers a working classification pipeline that predicts the health category of a food product from its nutritional profile, reaching a weighted F1-score of 0.999 with a tuned Decision Tree on the test set. The pipeline — cleaning, EDA, preprocessing, multi-model comparison, tuning, and a usable interactive interface — represents a complete, well-structured ML workflow, and the strong separability of the classes made the Decision Tree a particularly effective choice for this dataset.

---

## 13. Recommendations & Future Work

- Evaluate boosting algorithms (XGBoost, LightGBM, CatBoost) for additional comparison.
- Engineer additional features (nutrient ratios, composite health indices).
- Migrate the interface from `ipywidgets` to a standalone web application (Streamlit / Flask / FastAPI) for easier deployment and access outside the notebook environment.
- Extend cross-validation to all models, not just the Decision Tree.
- Explore additional class-balancing techniques (e.g. SMOTE) given the 68.85% / 22.69% / 8.46% class split.
