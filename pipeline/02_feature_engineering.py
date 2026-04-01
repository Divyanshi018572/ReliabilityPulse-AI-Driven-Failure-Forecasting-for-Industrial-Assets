import pandas as pd
import numpy as np
import os
import sys

# Add the project root to sys.path to import path_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def perform_feature_engineering():
    # Load raw data
    raw_path = path_utils.get_raw_data_path('ai4i2020.csv')
    if not os.path.exists(raw_path):
        print(f"Error: Raw dataset not found at {raw_path}")
        return

    df = pd.read_csv(raw_path)
    print("Raw data loaded.")

    # 1. Temperature Difference (Process - Air)
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']

    # 2. Power (Torque * angular speed in rad/s)
    # Angular speed = RPM * 2 * PI / 60
    df['power'] = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * np.pi / 60)

    # 3. Tool Wear * Torque (Mechanical stress indicator)
    df['tool_wear_torque'] = df['Tool wear [min]'] * df['Torque [Nm]']

    # 4. Ordinal Encoding for Type (L < M < H quality)
    type_map = {'L': 0, 'M': 1, 'H': 2}
    df['Type'] = df['Type'].map(type_map)

    # 5. Drop Data Leakage and Unnecessary columns
    # Sub-labels (TWF, HDF, PWF, OSF, RNF) indicate the cause of failure, which is leakage for binary classification
    cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df = df.drop(columns=cols_to_drop)

    # Save processed features
    processed_path = path_utils.get_processed_data_path('features.csv')
    df.to_csv(processed_path, index=False)
    print(f"Feature engineering complete. File saved to {processed_path}")
    print(f"Columns in processed data: {df.columns.tolist()}")

if __name__ == "__main__":
    perform_feature_engineering()
