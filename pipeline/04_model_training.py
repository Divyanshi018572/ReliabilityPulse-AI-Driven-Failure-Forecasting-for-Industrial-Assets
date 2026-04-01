import numpy as np
import os
import sys
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Add the project root to sys.path to import path_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def train_models():
    # Load preprocessed arrays
    preprocessed_path = path_utils.get_processed_data_path('preprocessed_data.pkl')
    if not os.path.exists(preprocessed_path):
        print(f"Error: Preprocessed data not found at {preprocessed_path}")
        return

    data = joblib.load(preprocessed_path)
    X_train = data['X_train']
    y_train = data['y_train']
    print("Preprocessed data loaded.")

    # 1. Isolation Forest (Unsupervised Baseline)
    # Contamination should be roughly equal to the failure rate in original data (~3.5%)
    print("Training Isolation Forest...")
    clf_iso = IsolationForest(contamination=0.035, random_state=42)
    clf_iso.fit(X_train)
    joblib.dump(clf_iso, path_utils.get_model_path('isolation_forest.pkl'))

    # 2. Logistic Regression (Baseline)
    print("Training Logistic Regression...")
    clf_lr = LogisticRegression(random_state=42, max_iter=1000)
    clf_lr.fit(X_train, y_train)
    joblib.dump(clf_lr, path_utils.get_model_path('logistic_regression.pkl'))

    # 3. Support Vector Machine (SVM)
    print("Training SVM...")
    clf_svm = SVC(kernel='rbf', probability=True, random_state=42)
    clf_svm.fit(X_train, y_train)
    joblib.dump(clf_svm, path_utils.get_model_path('svm_model.pkl'))

    # 4. Random Forest (Robust Ensemble)
    print("Training Random Forest...")
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(X_train, y_train)
    joblib.dump(clf_rf, path_utils.get_model_path('random_forest.pkl'))

    # 5. Decision Tree (Interpretable)
    print("Training Decision Tree...")
    clf_dt = DecisionTreeClassifier(random_state=42)
    clf_dt.fit(X_train, y_train)
    joblib.dump(clf_dt, path_utils.get_model_path('decision_tree.pkl'))

    # 6. XGBoost (Best Performer)
    print("Training XGBoost with GridSearch...")
    # scale_pos_weight is for imbalanced data, but since we used SMOTE, it's 1.0 (balanced)
    # However, I'll tune some key parameters
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_xgb = grid_search.best_estimator_
    print(f"Best XGBoost Params: {grid_search.best_params_}")
    joblib.dump(best_xgb, path_utils.get_model_path('xgboost_model.pkl'))

    print("All models trained and saved in 'models/' directory.")

if __name__ == "__main__":
    train_models()
