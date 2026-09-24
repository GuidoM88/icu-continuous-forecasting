from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """AUROC / AUPRC, the two metrics reported in the PhysioNet-2012 /
    Latent-ODE / Neural-CDE mortality-prediction literature. Guards against
    the degenerate single-class case (can happen on tiny synthetic batches)."""
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }
