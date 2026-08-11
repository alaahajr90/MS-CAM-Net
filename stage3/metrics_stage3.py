"""Classification metrics used for MS-CAM-Net Stage 3."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, probabilities) -> dict[str, float]:
    """Compute Accuracy, weighted Precision/Recall/F1, and AUC.

    Classification MAE is intentionally not computed because it is not part of
    the final manuscript evaluation protocol.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probabilities = np.asarray(probabilities)

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred) * 100.0),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0),
    }

    try:
        if probabilities.ndim != 2:
            raise ValueError("Probabilities must be a two-dimensional array.")
        if probabilities.shape[1] == 2:
            result["auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
        else:
            result["auc"] = float(
                roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
            )
    except ValueError:
        result["auc"] = float("nan")
    return result


def confusion(y_true, y_pred, labels) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist()
