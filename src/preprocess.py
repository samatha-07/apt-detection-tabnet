import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from src.config import MODEL_DIR, RANDOM_STATE

def preprocess_data(df, target_col="Label"):

    df = df.copy()

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[target_col])
    X = df.drop(columns=[target_col])

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save encoders
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    return X_scaled, y, target_encoder


def apply_smote(X, y):
    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(X, y)


def apply_pca(X, n_components=0.95):
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    joblib.dump(pca, os.path.join(MODEL_DIR, "pca_model.pkl"))
    return X_pca, pca
