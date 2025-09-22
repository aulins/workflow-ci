import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train(data_path: Path):
    # Load dataset
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].astype(int).values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("diabetes-basic")

    mlflow.sklearn.autolog()

    with mlflow.start_run():   # TANPA run_name
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)


    print(f"Training selesai ✅ | Akurasi: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        # dataset ada di luar folder MLProject
        default=str(Path(__file__).resolve().parent.parent / "dataset_preprocessing" / "diabetes_preprocessed.csv"),
        help="Path ke dataset",
    )
    args = parser.parse_args()
    train(Path(args.data))
