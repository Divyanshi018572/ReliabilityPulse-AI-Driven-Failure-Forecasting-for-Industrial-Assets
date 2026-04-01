import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score

# Add the project root to sys.path to import path_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def evaluate_models():
    # Load data
    preprocessed_path = path_utils.get_processed_data_path('preprocessed_data.pkl')
    if not os.path.exists(preprocessed_path):
        print(f"Error: Preprocessed data not found at {preprocessed_path}")
        return

    data = joblib.load(preprocessed_path)
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']
    print("Data loaded for evaluation.")

    # Load models
    models_to_eval = {
        'Logistic Regression': joblib.load(path_utils.get_model_path('logistic_regression.pkl')),
        'SVM': joblib.load(path_utils.get_model_path('svm_model.pkl')),
        'Random Forest': joblib.load(path_utils.get_model_path('random_forest.pkl')),
        'Decision Tree': joblib.load(path_utils.get_model_path('decision_tree.pkl')),
        'XGBoost': joblib.load(path_utils.get_model_path('xgboost_model.pkl')),
        'Isolation Forest': joblib.load(path_utils.get_model_path('isolation_forest.pkl'))
    }

    plt.figure(figsize=(10, 8))
    
    results = []

    for name, model in models_to_eval.items():
        print(f"\nEvaluating {name}...")
        
        if name == 'Isolation Forest':
            # Isolation Forest returns -1 for anomaly, 1 for normal
            # Convert to 0 for normal, 1 for anomaly
            preds_raw = model.predict(X_test)
            y_pred = np.where(preds_raw == -1, 1, 0)
            
            # Anomaly score distribution
            scores = -model.decision_function(X_test) # Higher scores = more anomalous
            plt.figure(figsize=(8, 6))
            sns.histplot(scores, bins=50, kde=True, color='purple')
            plt.title('Anomaly Scores (Isolation Forest)')
            plt.savefig(path_utils.get_output_path('anomaly_scores.png'))
            plt.close()
        else:
            y_pred = model.predict(X_test)
            
            # ROC Curve components
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        # Metrics
        print(classification_report(y_test, y_pred))
        f1 = f1_score(y_test, y_pred)
        results.append({'Model': name, 'F1-Score': f1})

        # Confusion Matrix for the best model (XGBoost)
        if name == 'XGBoost':
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - XGBoost')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(path_utils.get_output_path('confusion_matrix_xgboost.png'))
            plt.close()

            # Feature Importance
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
            plt.figure(figsize=(10, 7))
            feat_imp.plot(kind='barh', color='teal')
            plt.title('Top 10 Feature Importances (XGBoost)')
            plt.savefig(path_utils.get_output_path('feature_importance.png'))
            plt.close()

    # Final ROC Plot formatting
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.savefig(path_utils.get_output_path('roc_curve_comparison.png'))
    plt.close()

    # Summary Table
    res_df = pd.DataFrame(results)
    print("\nModel Performance Summary (F1-Score):")
    print(res_df.to_string(index=False))

    print("\nEvaluation completed. All plots saved to 'outputs/' directory.")

if __name__ == "__main__":
    evaluate_models()
