import os
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from src.config import MODEL_DIR

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_baseline_tabnet():
    return TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        device_name=get_device(),
        verbose=1
    )


def create_optimized_tabnet():
    return TabNetClassifier(
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-2, weight_decay=1e-5),
        device_name=get_device(),
        verbose=1
    )


def save_model(model, filename):
    path = os.path.join(MODEL_DIR, filename)
    model.save_model(path)


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    clf = TabNetClassifier()
    clf.load_model(path + ".zip")
    return clf
