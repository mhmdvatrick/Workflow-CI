import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.sklearn.autolog(log_models=False, log_input_examples=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="WA_Fn-UseC_-Telco-Customer-Churn_preprocessing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folder_path = args.data_path

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder_path}")

    X_train = pd.read_csv(os.path.join(folder_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(folder_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(folder_path, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(folder_path, "y_test.csv")).squeeze()

    with mlflow.start_run(run_name="ci_rf_autolog"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="telco-churn-ci-model",
            input_example=X_train.head(3),
        )

    print("CI training selesai.")


if __name__ == "__main__":
    main()
