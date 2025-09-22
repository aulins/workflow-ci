import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split


def train(data_path: Path):
    # 1) Load dataset
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].astype(int).values

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("diabetes-basic")
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=False)

    with mlflow.start_run(run_name="logreg_baseline"):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        # 4) Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        fig_path = Path("confusion_matrix.png")
        plt.tight_layout()
        fig.savefig(fig_path)
        mlflow.log_artifact(str(fig_path))

    print("✅ Training selesai. Cek artefak & metrik di ./mlruns atau MLflow UI.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "dataset_preprocessing"
            / "diabetes_preprocessed.csv"
        ),
        help="Path ke CSV hasil preprocessing",
    )
    args = parser.parse_args()
    train(Path(args.data))
