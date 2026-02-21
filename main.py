import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocess import preprocess_data, apply_smote, apply_pca
from src.train import train_model
from src.evaluate import evaluate_model
from src.config import RANDOM_STATE, CKC_STAGES

def main():

    target_col = 'Label'

    try:
        df = load_data('dataset.csv')
    except:
        print("[WARNING] No dataset found. Generating synthetic data...")
        from sklearn.datasets import make_classification

        X_syn, y_syn = make_classification(
            n_samples=5000,
            n_features=25,
            n_classes=len(CKC_STAGES),
            n_informative=15,
            random_state=RANDOM_STATE
        )

        df = pd.DataFrame(X_syn, columns=[f'feature_{i}' for i in range(25)])
        df[target_col] = y_syn

    X_scaled, y, target_encoder = preprocess_data(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y
    )

    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    X_train_pca, pca_model = apply_pca(X_train_smote)
    X_test_pca = pca_model.transform(X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_pca, y_train_smote,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_smote
    )

    baseline_model = train_model(X_tr, y_tr, X_val, y_val, 'baseline')
    optimized_model = train_model(X_tr, y_tr, X_val, y_val, 'optimized')

    evaluate_model(baseline_model, X_test_pca, y_test, "Baseline")
    evaluate_model(optimized_model, X_test_pca, y_test, "Optimized")

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
