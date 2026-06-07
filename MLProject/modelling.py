import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_DIR = Path("breast_cancer_preprocessing")


def main(n_estimators=150, max_depth=10):
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["diagnosis"]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["diagnosis"]

    with mlflow.start_run(run_name="ci_retraining_random_forest"):
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=None if str(max_depth).lower() == "none" else int(max_depth),
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        mlflow.log_param("n_estimators", int(n_estimators))
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_test.head(3))
        print("CI retraining selesai.")


if __name__ == "__main__":
    main(
        n_estimators=os.getenv("N_ESTIMATORS", 150),
        max_depth=os.getenv("MAX_DEPTH", 10),
    )
