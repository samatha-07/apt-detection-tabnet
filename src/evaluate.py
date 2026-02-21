import numpy as np
from sklearn.metrics import accuracy_score
import os
from src.config import MODEL_DIR

def evaluate_model(model, X_test, y_test, model_name='Model'):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{model_name} Accuracy: {acc:.4f}")
    return acc
