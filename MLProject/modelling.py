import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--test_size",    type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

mlflow.set_tracking_uri(os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/hajuenter/ach_bahrul_heart_disease_mlflow.mlflow"
))
mlflow.set_experiment("Heart Disease CI Three")


def train():
    df = pd.read_csv("heart_preprocessing/heart_clean.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    with mlflow.start_run():
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("test_size",    args.test_size)
        mlflow.log_param("random_state", args.random_state)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average="weighted"))
        mlflow.log_metric("recall",    recall_score(y_test, y_pred, average="weighted"))
        mlflow.log_metric("f1_score",  f1_score(y_test, y_pred, average="weighted"))

        mlflow.sklearn.log_model(model, "model")

        if os.path.exists("outputs"):
            shutil.rmtree("outputs")
        mlflow.sklearn.save_model(model, "outputs")

        run_id = mlflow.active_run().info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        print(f"Training selesai | run_id: {run_id}")


if __name__ == "__main__":
    train()