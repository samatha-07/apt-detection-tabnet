import os
import pandas as pd
from src.config import RAW_DATA_DIR

def load_data(file_name="dataset.csv"):
    file_path = os.path.join(RAW_DATA_DIR, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            "Place dataset inside data/raw/"
        )

    print(f"[INFO] Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    print(f"[INFO] Dataset shape: {df.shape}")
    return df
