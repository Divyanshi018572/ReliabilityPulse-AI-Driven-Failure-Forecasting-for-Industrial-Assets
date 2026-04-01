**PROJECT INSTRUCTION FILE**

**Predictive Maintenance**

Machine Learning Project  |  Classification + Anomaly Detection  |  Manufacturing Domain

|<p>Dataset</p><p>**AI4I 2020 Predictive Maintenance (Kaggle)**</p>|<p>Rows</p><p>**10,000 records**</p>|<p>Difficulty</p><p>**Easy**</p>|<p>Target Metric</p><p>**F1-Score (critical) 88–95%**</p>|
| :-: | :-: | :-: | :-: |


# **1. Project Overview**
Predictive maintenance (PdM) uses sensor data and machine telemetry to predict equipment failures before they occur, allowing scheduled maintenance instead of reactive repairs. This reduces unplanned downtime, extends machine lifespan, and lowers maintenance costs significantly.

|**Real-World Use Case**|
| :- |
|Manufacturing companies like Bosch, Siemens, and Tata Steel deploy PdM systems on CNC machines, turbines, and assembly lines. A single hour of unplanned downtime on an auto assembly line can cost ₹1–5 crore. PdM models monitoring temperature, torque, and vibration can detect anomalies 24–72 hours before mechanical failure.|

# **2. Dataset Details**
**Source**

- Name: AI4I 2020 Predictive Maintenance Dataset
- Platform: Kaggle — https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
- Format: CSV — single file
- License: Public / Open Use

**Dataset Statistics**

|**Property**|**Value**|
| :- | :- |
|Total Rows|10,000 machine readings|
|Total Columns|14 features|
|Target Column|Machine failure (0 = no failure, 1 = failure)|
|Class Distribution|~96.5% no failure, ~3.5% failure (highly imbalanced)|
|Missing Values|None|
|Data Types|Mix of numeric and categorical|

**Key Features**

- UDI — unique identifier (drop before modeling)
- Product ID — product serial with quality type prefix (L/M/H) — extract quality type
- Type — product quality: L (Low), M (Medium), H (High) — encode as ordinal
- Air temperature [K] — ambient air temperature in Kelvin
- Process temperature [K] — machine process temperature in Kelvin
- Rotational speed [rpm] — motor rotational speed
- Torque [Nm] — applied torque
- Tool wear [min] — cumulative tool usage time in minutes
- Machine failure — **TARGET**: 1 if any failure occurred
- TWF — Tool Wear Failure (sub-label)
- HDF — Heat Dissipation Failure (sub-label)
- PWF — Power Failure (sub-label)
- OSF — Overstrain Failure (sub-label)
- RNF — Random Failure (sub-label)

|**Multi-Label Insight**|
| :- |
|The dataset has 5 specific failure mode sub-labels (TWF, HDF, PWF, OSF, RNF) in addition to the overall Machine failure target. For the main model, predict Machine failure. For advanced analysis, build separate models for each failure mode or use multi-label classification.|

# **3. Step-by-Step Workflow**
## **Step 1 — Environment Setup**
Install the required Python libraries before starting:

|pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn|
| :- |

## **Step 2 — Load & Explore Data (EDA)**
1. Load CSV: df = pd.read\_csv('ai4i2020.csv')
2. Check shape, dtypes, nulls — confirm no missing values
3. Plot failure distribution — confirm ~3.5% failure rate (highly imbalanced)
4. Plot failure rate by product Type (L/M/H)
5. Plot distributions of temperature, torque, rotational speed, tool wear
6. Box plots: compare sensor readings for failure vs non-failure cases
7. Correlation heatmap for numeric features
8. Plot failure count by each sub-label (TWF, HDF, PWF, OSF, RNF)

|**Key EDA Finding**|
| :- |
|Tool wear > 200 min combined with high torque is the strongest predictor of failure. Heat Dissipation Failures (HDF) occur when temperature difference between process and air temperature is < 8.6 K. Power Failures (PWF) occur when power (torque × rotational speed) falls outside 3500–9000 W range. Engineering these derived features significantly improves model performance.|

## **Step 3 — Feature Engineering**
Engineer domain-informed features before preprocessing:

1. temp\_diff = df['Process temperature [K]'] - df['Air temperature [K]'] (HDF signal)
2. power = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * 3.14159 / 60) (PWF signal — power in Watts)
3. tool\_wear\_torque = df['Tool wear [min]'] * df['Torque [Nm]'] (OSF/TWF signal)
4. Extract quality type: df['Quality'] = df['Product ID'].str[0] → L=0, M=1, H=2 (ordinal encoding)
5. Drop: UDI, Product ID, TWF, HDF, PWF, OSF, RNF (sub-labels — data leakage for main target)

## **Step 4 — Data Preprocessing**
1. Encode Type column: map({'L': 0, 'M': 1, 'H': 2}) — ordinal makes sense here (quality order)
2. Scale numeric features (Air temp, Process temp, RPM, Torque, Tool wear, engineered features) using StandardScaler — required for SVM and Isolation Forest
3. Split: X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.2, random\_state=42, stratify=y)
4. Confirm class distribution in train and test sets

## **Step 5 — Handle Class Imbalance (Critical)**
With only ~3.5% failure rate, imbalance handling is essential:

- **Option A — SMOTE**: from imblearn.over\_sampling import SMOTE — generate synthetic failure samples (apply only on training data, AFTER split)
- **Option B — class\_weight='balanced'**: automatic weight adjustment in sklearn models
- **Option C — Threshold tuning**: lower classification threshold from 0.5 to 0.3 to maximize recall on failures
- **Recommended**: Use SMOTE for XGBoost + threshold tuning for final deployment

|**Critical: Recall is the Priority Metric**|
| :- |
|In predictive maintenance, a missed failure (False Negative) causes unplanned downtime and equipment damage. A false alarm (False Positive) triggers an unnecessary inspection — costly but not catastrophic. Always optimize for Recall > 85% on the failure class. Use F1-Score as the primary tuning metric, never accuracy.|

## **Step 6 — Model Building**

|**Model**|**When to Use**|**Expected F1 (Failure)**|
| :- | :- | :- |
|Logistic Regression|Baseline, fast, interpretable|55 – 65%|
|Random Forest|Handles class imbalance well with balanced weights|75 – 82%|
|XGBoost|Best overall performer for this dataset|80 – 88%|
|SVM (RBF kernel)|Works well on small-medium sensor datasets|72 – 80%|
|Isolation Forest|Anomaly detection — unsupervised baseline|60 – 70% (approx.)|

Recommended order: Isolation Forest for anomaly baseline → Logistic Regression → SVM → XGBoost as final classifier.

**Isolation Forest usage (anomaly detection approach):**
```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.035, random_state=42)
iso.fit(X_train)
preds = iso.predict(X_test)  # -1 = anomaly (potential failure), 1 = normal
```

## **Step 7 — Hyperparameter Tuning**
1. Use GridSearchCV or RandomizedSearchCV with cv=5
2. XGBoost key params: n\_estimators (100–400), max\_depth (3–7), learning\_rate (0.01–0.2), scale\_pos\_weight (set to ratio of negatives/positives ≈ 27 for imbalanced data)
3. SVM key params: C (0.1–100), gamma ('scale', 'auto', 0.001–0.1), kernel ('rbf', 'poly')
4. Use scoring='f1' as primary metric — not 'accuracy'

## **Step 8 — Evaluate the Model**

|**Metric**|**What it Measures**|**Target Value**|
| :- | :- | :- |
|Accuracy|Overall correct predictions|> 96% (easy due to imbalance — not reliable)|
|Precision (Failure)|Of predicted failures, how many were actual|> 75%|
|Recall (Failure)|Of actual failures, how many did we catch|> 85%|
|F1-Score (Failure)|Harmonic mean — primary metric|88 – 95%|
|AUC-ROC|Separation between classes|> 0.90|
|Confusion Matrix|Full TP/TN/FP/FN breakdown|Always visualize|

# **4. Feature Importance**

|**Rank**|**Feature**|**Importance Level**|**Business Insight**|
| :- | :- | :- | :- |
|1|tool\_wear\_torque (engineered)|Very High|Combined stress = primary failure driver|
|2|Tool wear [min]|Very High|Aging tools fail more — schedule replacements|
|3|Torque [Nm]|High|Overload indicator|
|4|temp\_diff (engineered)|High|Low temp diff = heat dissipation failure risk|
|5|power (engineered)|High|Out-of-range power = motor failure|
|6|Rotational speed [rpm]|Medium-High|Speed anomalies precede mechanical failures|
|7|Process temperature [K]|Medium|High process temp accelerates wear|
|8|Type (Quality)|Medium|Low-quality products run hotter, fail more|
|9|Air temperature [K]|Low-Medium|Ambient temp affects heat dissipation|

# **5. Project Structure**

```
04_predictive_maintenance/
├── data/
│   ├── raw/ai4i2020.csv
│   └── processed/features.csv
├── models/
│   ├── xgboost_model.pkl
│   └── isolation_forest.pkl
├── pipeline/
│   ├── 01_eda.py
│   ├── 02_feature_engineering.py
│   ├── 03_preprocessing.py
│   ├── 04_model_training.py
│   └── 05_evaluation.py
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── anomaly_scores.png
├── app.py
├── path_utils.py
└── README.md
```

**Pipeline File Descriptions:**

| File | Purpose |
| :- | :- |
| 01\_eda.py | Load data, plot distributions, failure rates, correlations, sub-label analysis |
| 02\_feature\_engineering.py | Create temp\_diff, power, tool\_wear\_torque, encode Type, drop leakage columns |
| 03\_preprocessing.py | Scale features, apply SMOTE on train set, save processed arrays |
| 04\_model\_training.py | Train Isolation Forest, Logistic Regression, SVM, XGBoost — save models |
| 05\_evaluation.py | Generate all metrics, confusion matrix, ROC curve, feature importance plots |

# **6. Expected Results Summary**

|**Metric**|**Baseline (Logistic Reg.)**|**Best Model (XGBoost + SMOTE)**|
| :- | :- | :- |
|Accuracy|> 96%|> 97%|
|Precision (Failure)|55 – 65%|75 – 85%|
|Recall (Failure)|60 – 70%|85 – 92%|
|F1-Score (Failure)|57 – 67%|80 – 88%|
|AUC-ROC|0.82 – 0.87|0.91 – 0.95|

# **7. Common Mistakes to Avoid**
- Using accuracy as the primary metric — with 96.5% no-failure, predicting all 'no failure' gives 96.5% accuracy but is completely useless
- Including sub-label columns (TWF, HDF, PWF, OSF, RNF) as features — they directly encode failure causes and cause severe data leakage
- Applying SMOTE before the train/test split — synthetic samples from test data leak into training
- Forgetting scale\_pos\_weight in XGBoost — set to ~27 (ratio of negative to positive) for imbalanced data
- Not engineering derived features (temp\_diff, power, tool\_wear\_torque) — raw features alone miss key failure physics
- Dropping Type column — product quality type has meaningful impact on failure rate

# **8. Recommended Tools & Libraries**

|**Library**|**Purpose**|
| :- | :- |
|pandas|Data loading, feature engineering|
|numpy|Numerical operations, power calculation|
|scikit-learn|Preprocessing, SVM, Isolation Forest, metrics|
|xgboost|Best classifier — handles imbalance with scale\_pos\_weight|
|imbalanced-learn|SMOTE for oversampling minority failure class|
|matplotlib / seaborn|EDA plots, confusion matrix heatmap, ROC curve|
|joblib|Save and load trained models|

# **9. Project Deliverables Checklist**
- pipeline/ folder with 5 modular .py files (EDA → feature engineering → preprocessing → training → evaluation)
- Trained XGBoost model + Isolation Forest saved as .pkl using joblib
- Classification Report + Confusion Matrix visualization
- ROC Curve comparing all models
- Feature Importance bar chart (top 9 features including engineered)
- Anomaly score distribution plot (Isolation Forest)
- README.md explaining failure modes and prediction threshold choice
- Streamlit app (app.py) for live failure risk prediction — user inputs sensor readings (temp, RPM, torque, tool wear, quality type), model returns failure probability with risk level (Low/Medium/High/Critical), top contributing factors, and recommended maintenance action

Predictive Maintenance  |  ML Project Instruction File  |  Classification + Anomaly Detection Project #4
