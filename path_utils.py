import os

# Define base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

def get_raw_data_path(filename='ai4i2020.csv'):
    return os.path.join(DATA_RAW_DIR, filename)

def get_processed_data_path(filename='features.csv'):
    return os.path.join(DATA_PROCESSED_DIR, filename)

def get_model_path(filename):
    return os.path.join(MODELS_DIR, filename)

def get_output_path(filename):
    return os.path.join(OUTPUTS_DIR, filename)

# Ensure directories exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
