import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib
import os
from src.config import MODEL_DIR, RANDOM_STATE

def preprocess_data(df, target_col='Label'):
    print("[INFO] Preprocessing data...")

    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    label_encoders = {}
    for col in df_clean.select_dtypes(include=['object']).columns:
        if col != target_col:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le

    target_le = LabelEncoder()
    y = target_le.fit_transform(df_clean[target_col].astype(str))
    X = df_clean.drop(columns=[target_col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(label_encoders, os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    joblib.dump(target_le, os.path.join(MODEL_DIR, 'target_encoder.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    return X_scaled, y, target_le

def apply_smote(X, y):
    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(X, y)

def apply_pca(X, n_components=0.95):
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    joblib.dump(pca, os.path.join(MODEL_DIR, 'pca_model.pkl'))
    return X_pca, pca
