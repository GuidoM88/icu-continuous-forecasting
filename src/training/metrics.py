from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Modern discrimination metrics used throughout the project."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def event1_score(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    """PhysioNet/CinC 2012 Event 1: min(sensitivity, positive predictivity)."""
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(prediction, dtype=int)
    if y.shape != pred.shape:
        raise ValueError("y_true and prediction must have the same shape")
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    return {"event1": min(sensitivity, ppv), "sensitivity": sensitivity, "ppv": ppv}


def select_event1_threshold(y_true: np.ndarray, risk: np.ndarray) -> float:
    """Choose an Event-1 threshold on development data only.

    Ties are resolved deterministically by choosing the threshold closest to 0.5.
    Never call this on Set B/C outcomes in a competition-faithful experiment.
    """
    y = np.asarray(y_true, dtype=int)
    risk = np.asarray(risk, dtype=float)
    candidates = np.unique(np.r_[0.0, risk, 1.0])
    scored = []
    for threshold in candidates:
        score = event1_score(y, risk >= threshold)["event1"]
        scored.append((score, -abs(float(threshold) - 0.5), float(threshold)))
    return max(scored)[2]


def event2_score(y_true: np.ndarray, risk: np.ndarray, n_groups: int = 10) -> float:
    """PhysioNet/CinC 2012 Event 2: range-normalized Hosmer-Lemeshow H/D.

    Implements the published Challenge definition, including +0.001 in each
    H-statistic denominator. For the official 4,000-record sets, the ten
    groups contain 400 records each.
    """
    y = np.asarray(y_true, dtype=float)
    risk = np.asarray(risk, dtype=float)
    if y.shape != risk.shape:
        raise ValueError("y_true and risk must have the same shape")
    if y.size < n_groups:
        raise ValueError(f"need at least {n_groups} observations")
    if np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("risk estimates must lie in [0, 1]")

    order = np.argsort(risk, kind="mergesort")
    groups = np.array_split(order, n_groups)
    h = 0.0
    mean_risks = []
    for ids in groups:
        n = len(ids)
        observed = float(y[ids].sum())
        pi = float(risk[ids].mean())
        expected = n * pi
        h += (observed - expected) ** 2 / (n * pi * (1.0 - pi) + 0.001)
        mean_risks.append(pi)

    spread = mean_risks[-1] - mean_risks[0]
    return float("inf") if spread <= 0 else float(h / spread)


def challenge_metrics(y_true: np.ndarray, risk: np.ndarray, threshold: float) -> dict:
    """AUROC/AUPRC plus the two official 2012 Challenge metrics."""
    risk = np.asarray(risk, dtype=float)
    out = classification_metrics(np.asarray(y_true), risk)
    out.update(event1_score(np.asarray(y_true), risk >= threshold))
    out["threshold"] = float(threshold)
    out["event2"] = event2_score(np.asarray(y_true), risk)
    return out
