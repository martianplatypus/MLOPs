import os
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

load_dotenv()

EXPECTED_ACCURACY = float(os.getenv("EXPECTED_ACCURACY", 0.85))


def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }

    return metrics


def check_accuracy(metrics):
    return metrics["accuracy"] >= EXPECTED_ACCURACY
