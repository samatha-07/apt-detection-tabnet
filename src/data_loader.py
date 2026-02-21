import pandas as pd
import os
from src.config import RAW_DATA_DIR

def load_data(file_name="dataset.csv"):
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    print(f"[INFO] Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"[INFO] Data loaded. Shape: {df.shape}")
    return df
