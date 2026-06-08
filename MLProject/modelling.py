import argparse
import os

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Enable MLflow Autolog
mlflow.sklearn.autolog()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="WA_Fn-UseC_-Telco-Customer-Churn_preprocessing",
    )
    return parser.parse_args()


def main():

    args = parse_args()

    X_train = pd.read_csv(
        os.path.join(args.data_path, "X_train.csv")
    )

    X_test = pd.read_csv(
        os.path.join(args.data_path, "X_test.csv")
    )

    y_train = pd.read_csv(
        os.path.join(args.data_path, "y_train.csv")
    ).squeeze()

    y_test = pd.read_csv(
        os.path.join(args.data_path, "y_test.csv")
    ).squeeze()

    with mlflow.start_run(run_name="Telco-Churn-RandomForest"):

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(5),
        )

        print("=" * 60)
        print("Training selesai")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()