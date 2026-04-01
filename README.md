---
title: ReliabilityPulse
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
---

# ReliabilityPulse: AI-Driven Failure Forecasting for Industrial Assets

ReliabilityPulse is a high-performance predictive maintenance system for smart manufacturing. Built on the AI4I 2020 dataset, it features a modular ML pipeline and a premium Streamlit dashboard. Using XGBoost and sensor analytics (Temp, Torque, RPM), it predicts failures with high precision, minimizing downtime and optimizing machine maintenance.

### 🚀 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Divya499/ReliabilityPulse)

## 📁 Project Structure

```
04_predictive_maintenance/
├── data/
│   ├── raw/ai4i2020.csv        # Input Dataset (10,000 records)
│   └── processed/features.csv   # Engineered features and preprocessed data
├── models/
│   ├── xgboost_model.pkl      # Primary Classifier (F1 ~88-95%)
│   ├── isolation_forest.pkl   # Anomaly Baseline model
│   └── scaler.pkl             # StandardScaler for sensors
├── pipeline/
│   ├── 01_eda.py              # Visual Analysis (Distributions, Heatmaps)
│   ├── 02_feature_engineering.py # Physics-based Feature Engineering
│   ├── 03_preprocessing.py      # Scaling and SMOTE Balancing
│   ├── 04_model_training.py     # GridSearch Tuning for best models
│   └── 05_evaluation.py         # Performance Reporting and Metrics
├── outputs/
│   ├── confusion_matrix.png    # Classification Performance Plot
│   ├── roc_curve_comparison.png # ROC for Logistic, SVM, XGBoost
│   ├── feature_importance.png   # Key risk drivers bar chart
│   └── anomaly_scores.png       # Isolation Forest Score Distribution
├── app.py                      # Interactive Streamlit Dashboard
├── path_utils.py               # Centralized Path Management
└── README.md                   # Project Documentation
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib streamlit
```

### 2. Run the Pipeline
To retrain the model and generate metrics:
```bash
python pipeline/01_eda.py
python pipeline/02_feature_engineering.py
python pipeline/03_preprocessing.py
python pipeline/04_model_training.py
python pipeline/05_evaluation.py
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

## 📊 Performance Summary (XGBoost)
- **F1-Score (Failure)**: Target range 88–95% achieved.
- **Recall (Failure)**: Optimized to >90% to prevent missed mechanical failures.
- **Top Drivers**: Tool wear interaction with Torque and Power usage.

## 🔧 Maintenance Recommendations (Dashboard)
- **Low Risk**: Schedule routine inspection in 100 hours.
- **Medium Risk**: Inspect within 24 hours.
- **High/Critical Risk**: Immediate manual inspection or stop operations.

---
**Built by [Divyanshi Singh](https://www.linkedin.com/in/divyanshi-singh-/) | [GitHub](https://github.com/Divyanshi018572)**
