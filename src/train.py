from src.model import create_baseline_tabnet, create_optimized_tabnet, save_model
from src.config import MAX_EPOCHS, PATIENCE, BATCH_SIZE

def train_model(X_train, y_train, X_valid, y_valid, model_type='baseline'):

    if model_type == 'baseline':
        clf = create_baseline_tabnet()
    else:
        clf = create_optimized_tabnet()

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE
    )

    save_model(clf, f'tabnet_{model_type}')
    return clf
