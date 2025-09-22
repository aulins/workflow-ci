# modelling.py
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def train(data_path: Path):
    # 1) Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].astype(int).values

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Set MLflow lokal (folder mlruns akan dibuat otomatis)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("diabetes-basic")

    # 4) Autolog ON
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=False)

    # 5) Train + log manual untuk metrik utama (agar jelas di UI)
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

        # 6) Log artefak: confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        fig_path = Path("confusion_matrix.png")
        plt.tight_layout()
        fig.savefig(fig_path)
        mlflow.log_artifact(str(fig_path))

    print("Training selesai. Cek folder ./mlruns atau buka MLflow UI.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=str(
            Path(__file__).with_name("dataset_preprocessing") / "diabetes_preprocessed.csv"
        ),
        help="Path ke CSV hasil preprocessing",
    )
    args = parser.parse_args()
    train(Path(args.data))
