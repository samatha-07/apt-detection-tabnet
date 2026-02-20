import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model directory
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create directories if not exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Kill Chain stages
CKC_STAGES = [
    "Benign",
    "Reconnaissance",
    "Initial Access",
    "Command & Control",
    "Data Exfiltration"
]

RANDOM_STATE = 42

# Training settings
BATCH_SIZE = 1024
MAX_EPOCHS = 50
PATIENCE = 10
