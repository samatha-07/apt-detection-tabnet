from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import os
from src.config import MODEL_DIR

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def create_baseline_tabnet():
    return TabNetClassifier(device_name=get_device(), verbose=1)

def create_optimized_tabnet():
    return TabNetClassifier(
        n_d=64, n_a=64, n_steps=5,
        device_name=get_device(),
        verbose=1
    )

def save_model(model, filename):
    path = os.path.join(MODEL_DIR, filename)
    model.save_model(path)
