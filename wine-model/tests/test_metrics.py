import pytest
from wine_model.metrics import calculate_metrics, check_accuracy


def test_calculate_metrics():
    y_true = [0, 1, 2, 2, 1]
    y_pred = [0, 2, 2, 2, 1]
    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.6
    assert metrics["precision"] == pytest.approx(0.73, 0.01)
    assert metrics["recall"] == pytest.approx(0.6, 0.01)
    assert metrics["f1_score"] == pytest.approx(0.62, 0.01)


def test_check_accuracy():
    metrics = {
        "accuracy": 0.9,
        "precision": 0.9,
        "recall": 0.9,
        "f1_score": 0.9,
    }
    assert check_accuracy(metrics) == True

    metrics["accuracy"] = 0.8
    assert check_accuracy(metrics) == False
