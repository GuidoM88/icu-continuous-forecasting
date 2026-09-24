"""
Generates fake-but-structurally-realistic PatientRecord objects, matching
the exact shape the PhysioNet parser produces. Two uses:

1. Smoke-testing the pipeline / models without the real (~1.2GB) download.
2. Letting you run `python -m src.training.train --synthetic` on day one,
   before you've even pulled the real dataset, to check the whole loop
   works end to end.

The generated series are NOT clinically meaningful -- just irregularly
sampled, partially observed multivariate noise with a signal correlated to
the label, enough to exercise every code path (missingness, variable
sequence density, static features, class imbalance).
"""
from __future__ import annotations

import random
from typing import List

import numpy as np

from .physionet import GENERAL_DESCRIPTORS, PatientRecord

SYNTHETIC_VARS = [
    "HR", "SysABP", "DiasABP", "MAP", "Temp", "RespRate", "SaO2", "GCS",
    "Glucose", "Creatinine", "Urine", "WBC", "K", "Na", "pH", "Lactate",
]


def make_synthetic_records(
    n: int = 400, max_hours: float = 48.0, death_rate: float = 0.15, seed: int = 0
) -> List[PatientRecord]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    records = []

    for i in range(n):
        label = 1 if rng.random() < death_rate else 0
        # sicker (label=1) patients get noisier vitals and are observed more often
        severity = 1.0 if label else 0.0

        static = {
            "Age": float(np_rng.integers(20, 90)),
            "Gender": float(rng.randint(0, 1)),
            "Height": float(np_rng.normal(170, 10)),
            "ICUType": float(rng.randint(1, 4)),
            "Weight": float(np_rng.normal(75, 15)),
        }

        values = {}
        for var in SYNTHETIC_VARS:
            base_rate = np_rng.uniform(0.1, 0.6) + 0.3 * severity  # obs / hour
            t = 0.0
            pts = []
            base_level = np_rng.normal(0, 1)
            while t < max_hours:
                t += np_rng.exponential(1.0 / max(base_rate, 1e-3))
                if t >= max_hours:
                    break
                drift = 0.03 * severity * t
                val = base_level + drift + np_rng.normal(0, 0.5 + 0.5 * severity)
                pts.append((round(t, 2), float(val)))
            if pts:
                values[var] = pts

        times = sorted({t for pts in values.values() for t, _ in pts})
        records.append(
            PatientRecord(
                record_id=f"SYN{i:05d}", static=static, times=times, values=values, label=label
            )
        )
    return records
