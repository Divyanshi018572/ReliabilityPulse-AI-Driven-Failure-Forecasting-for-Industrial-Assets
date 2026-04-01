import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the project root to sys.path to import path_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def perform_eda():
    # Load data
    raw_data_path = path_utils.get_raw_data_path('ai4i2020.csv')
    if not os.path.exists(raw_data_path):
        print(f"Error: Dataset not found at {raw_data_path}")
        return

    df = pd.read_csv(raw_data_path)
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(df.info())

    # 1. Failure Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Machine failure', data=df, palette='viridis')
    plt.title('Machine Failure Distribution (Target)')
    plt.savefig(path_utils.get_output_path('failure_distribution.png'))
    plt.close()

    # 2. Failure Rate by Product Type
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Type', y='Machine failure', data=df, palette='magma')
    plt.title('Failure Rate by Product Type (L/M/H)')
    plt.savefig(path_utils.get_output_path('failure_rate_by_type.png'))
    plt.close()

    # 3. Numeric Distributions
    numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 
                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[col], kde=True, color='teal')
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(path_utils.get_output_path('numeric_distributions.png'))
    plt.close()

    # 4. Box plots: compare sensor readings for failure vs non-failure
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='Machine failure', y=col, data=df, palette='Set2')
        plt.title(f'{col} vs Machine Failure')
    plt.tight_layout()
    plt.savefig(path_utils.get_output_path('sensor_boxplots.png'))
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_cols + ['Machine failure']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig(path_utils.get_output_path('correlation_heatmap.png'))
    plt.close()

    # 6. Sub-label Analysis (Failure Modes)
    sub_labels = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    plt.figure(figsize=(10, 6))
    df[sub_labels].sum().sort_values(ascending=False).plot(kind='bar', color='salmon')
    plt.title('Count of Each Failure Mode (Sub-labels)')
    plt.ylabel('Count')
    plt.savefig(path_utils.get_output_path('sub_label_counts.png'))
    plt.close()

    print("EDA completed and plots saved in 'outputs/' directory.")

if __name__ == "__main__":
    perform_eda()
