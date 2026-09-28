import numpy as np

from src.training.metrics import event1_score, event2_score, select_event1_threshold


def test_event1_is_min_sensitivity_and_ppv():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    m = event1_score(y, pred)
    assert m["sensitivity"] == 0.5
    assert m["ppv"] == 0.5
    assert m["event1"] == 0.5


def test_event1_threshold_is_valid():
    y = np.array([0, 0, 1, 1])
    risk = np.array([0.1, 0.2, 0.8, 0.9])
    t = select_event1_threshold(y, risk)
    assert 0.0 <= t <= 1.0
    assert event1_score(y, risk >= t)["event1"] == 1.0


def test_event2_is_finite_for_spread_risks():
    y = np.array(([0] * 8 + [1] * 2) * 10)
    risk = np.linspace(0.01, 0.99, len(y))
    score = event2_score(y, risk)
    assert np.isfinite(score)
    assert score >= 0.0
