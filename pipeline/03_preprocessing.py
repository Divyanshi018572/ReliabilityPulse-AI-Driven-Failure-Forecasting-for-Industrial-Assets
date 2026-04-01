import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Add the project root to sys.path to import path_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def perform_preprocessing():
    # Load feature-engineered data
    features_path = path_utils.get_processed_data_path('features.csv')
    if not os.path.exists(features_path):
        print(f"Error: Processed features not found at {features_path}")
        return

    df = pd.read_csv(features_path)
    print("Processed features loaded.")

    # Separate features and target
    X = df.drop(columns=['Machine failure'])
    y = df['Machine failure']

    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split completed. Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"Original failure distribution in training: {np.bincount(y_train)}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for use in the app
    joblib.dump(scaler, path_utils.get_model_path('scaler.pkl'))
    print("Scaler saved to models/scaler.pkl")

    # Apply SMOTE on Training data only
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"SMOTE completed. Resampled failure distribution: {np.bincount(y_train_resampled)}")

    # Save preprocessed components
    preprocessed_data = {
        'X_train': X_train_resampled,
        'X_test': X_test_scaled,
        'y_train': y_train_resampled,
        'y_test': y_test.values,
        'feature_names': X.columns.tolist()
    }
    
    joblib.dump(preprocessed_data, path_utils.get_processed_data_path('preprocessed_data.pkl'))
    print(f"Preprocessed arrays saved to {path_utils.get_processed_data_path('preprocessed_data.pkl')}")

if __name__ == "__main__":
    perform_preprocessing()
