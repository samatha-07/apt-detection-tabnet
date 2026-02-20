import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.config import MODEL_DIR

def evaluate_model(model, X_test, y_test, target_encoder, model_name):

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")

    print(f"{model_name} Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=target_encoder.classes_,
                yticklabels=target_encoder.classes_)
    plt.title(model_name)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{model_name}_cm.png"))
    plt.close()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
