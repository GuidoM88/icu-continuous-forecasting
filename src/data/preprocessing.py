"""
Turns a list of PatientRecord into batch tensors shared by every model
(GRU / GRU-D need delta-times; Neural CDE needs a spline-ready tensor;
Latent ODE needs a single query-time vector shared across the batch).

DESIGN CHOICE -- shared hourly grid.
Every patient's first 48h are binned onto the *same* reference grid
`t = 0, 1, 2, ..., 48` (configurable via `bin_size_hours`). Multiple raw
observations of one variable landing in the same bin are averaged. This
keeps every model exactly comparable, and -- crucially -- lets the Latent
ODE integrate the whole batch on one shared time vector instead of the
union-of-all-timestamps + gather-by-index bookkeeping the fully continuous
version needs. Missingness is *not* lost: within this grid, most
(patient, hour, variable) cells are still unobserved (mask = 0), so this
is still a genuine irregular / partially-observed setting, just discretised
at 1-hour resolution. This is a standard simplification in the ICU
time-series literature.

>>> NEXT STEP FOR YOU (a natural week-3 extension, see README): drop the
binning and integrate on the true union of observation times per batch.
The model code in src/models/ doesn't need to change, only this file and
the `times` field of Batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import torch

from .physionet import GENERAL_DESCRIPTORS, PatientRecord


def build_vocab(records: Sequence[PatientRecord]) -> List[str]:
    """Variable vocabulary is *built from the data*, not hard-coded: not
    every one of the ~37-42 documented Challenge variables necessarily
    appears in every release / subsample of the data."""
    vocab = set()
    for r in records:
        # Defensive filtering: malformed blank variable names must never
        # become model features, even if a PatientRecord was built elsewhere.
        vocab.update(v.strip() for v in r.values.keys() if v and v.strip())
    return sorted(vocab)


@dataclass
class Normalizer:
    mean: np.ndarray  # [D]
    std: np.ndarray  # [D]
    static_mean: np.ndarray  # [S]
    static_std: np.ndarray  # [S]

    @staticmethod
    def fit(values: np.ndarray, mask: np.ndarray, static: np.ndarray) -> "Normalizer":
        D = values.shape[-1]
        mean = np.zeros(D, dtype=np.float32)
        std = np.ones(D, dtype=np.float32)
        for d in range(D):
            obs = values[..., d][mask[..., d] > 0]
            if obs.size > 0:
                mean[d] = obs.mean()
                s = obs.std()
                std[d] = s if s > 1e-6 else 1.0
        static_mean = np.nanmean(static, axis=0)
        static_std = np.nanstd(static, axis=0)
        static_std = np.where(static_std > 1e-6, static_std, 1.0)
        static_mean = np.nan_to_num(static_mean, nan=0.0)
        return Normalizer(mean, std, static_mean.astype(np.float32), static_std.astype(np.float32))

    def apply(self, values: np.ndarray, static: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        values = (values - self.mean) / self.std
        static = (static - self.static_mean) / self.static_std
        static = np.nan_to_num(static, nan=0.0)
        return values, static


@dataclass
class Batch:
    times: torch.Tensor       # [T_bins]      hours, shared across the whole dataset
    values: torch.Tensor      # [B, T_bins, D] NaN where unobserved (post-normalisation)
    mask: torch.Tensor        # [B, T_bins, D] 1.0 observed, 0.0 otherwise
    delta_t: torch.Tensor     # [B, T_bins, D] hours since var last observed (GRU-D style)
    static: torch.Tensor      # [B, S]         already normalised, NaNs zero-filled
    labels: torch.Tensor      # [B]
    record_ids: List[str] = field(default_factory=list)

    def to(self, device):
        return Batch(
            times=self.times.to(device),
            values=self.values.to(device),
            mask=self.mask.to(device),
            delta_t=self.delta_t.to(device),
            static=self.static.to(device),
            labels=self.labels.to(device),
            record_ids=self.record_ids,
        )


def compute_delta_t(mask: np.ndarray) -> np.ndarray:
    """GRU-D style time-since-last-observation (Che et al. 2018), assuming a
    uniform 1-unit grid spacing (the grid IS the time axis here)."""
    B, T, D = mask.shape
    delta = np.zeros((B, T, D), dtype=np.float32)
    for t in range(1, T):
        observed_prev = mask[:, t - 1] > 0
        delta[:, t] = np.where(observed_prev, 1.0, 1.0 + delta[:, t - 1])
    return delta


def fill_forward(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Forward-fill each variable from its last observation; anything still
    unobserved at t=0 is filled with 0 (== the training-set mean, since
    values are normalised before this is called)."""
    B, T, D = values.shape
    out = values.copy()
    last = np.zeros((B, D), dtype=np.float32)
    seen = np.zeros((B, D), dtype=bool)
    for t in range(T):
        obs = mask[:, t] > 0
        cur = np.where(obs, np.nan_to_num(out[:, t], nan=0.0), last)
        out[:, t] = np.where(seen | obs, cur, 0.0)
        last = cur
        seen = seen | obs
    return out


class Physionet2012Dataset:
    """Bins raw PatientRecords onto a shared hourly grid. Call `.collate(idx)`
    with a list of indices to get a ready-to-train Batch."""

    def __init__(
        self,
        records: Sequence[PatientRecord],
        vocab: Sequence[str],
        static_names: Sequence[str] = tuple(GENERAL_DESCRIPTORS),
        max_hours: float = 48.0,
        bin_size_hours: float = 1.0,
        normalizer: "Normalizer | None" = None,
    ):
        # Keep unlabeled records: Set B/C must be transformable at inference
        # time without exposing their outcomes.
        self.records = list(records)
        self.vocab = list(vocab)
        self.static_names = list(static_names)
        self.var_index = {v: i for i, v in enumerate(self.vocab)}
        self.max_hours = max_hours
        self.bin_size_hours = bin_size_hours
        self.n_bins = int(max_hours // bin_size_hours) + 1
        self.times = np.arange(self.n_bins, dtype=np.float32) * bin_size_hours
        self.normalizer = normalizer

    def __len__(self):
        return len(self.records)

    def fit_normalizer(self) -> Normalizer:
        raw = [self._bin_one(i, normalize=False) for i in range(len(self))]
        values = np.stack([r[0] for r in raw])
        mask = np.stack([r[1] for r in raw])
        static = np.stack([r[2] for r in raw])
        self.normalizer = Normalizer.fit(values, mask, static)
        return self.normalizer

    def _bin_one(self, idx: int, normalize: bool = True):
        rec = self.records[idx]
        D = len(self.vocab)
        values = np.zeros((self.n_bins, D), dtype=np.float32)
        counts = np.zeros((self.n_bins, D), dtype=np.float32)
        mask = np.zeros((self.n_bins, D), dtype=np.float32)

        for var, pts in rec.values.items():
            j = self.var_index.get(var)
            if j is None:
                continue
            for t, v in pts:
                if t > self.max_hours:
                    continue
                b = int(round(t / self.bin_size_hours))
                b = min(b, self.n_bins - 1)
                values[b, j] += v
                counts[b, j] += 1.0

        observed = counts > 0
        values[observed] /= counts[observed]
        mask[observed] = 1.0

        static = np.array(
            [rec.static.get(name, np.nan) for name in self.static_names], dtype=np.float32
        )
        if normalize and self.normalizer is not None:
            values, static = self.normalizer.apply(values, static)
        label = np.nan if rec.label is None else float(rec.label)
        return values, mask, static, label

    def collate(self, indices: Sequence[int]) -> Batch:
        if self.normalizer is None:
            raise RuntimeError("Call fit_normalizer() on the *training* split first.")
        raw = [self._bin_one(i) for i in indices]
        values = np.stack([r[0] for r in raw])
        mask = np.stack([r[1] for r in raw])
        static = np.stack([r[2] for r in raw])
        labels = np.array([r[3] for r in raw], dtype=np.float32)
        delta_t = compute_delta_t(mask)
        record_ids = [self.records[i].record_id for i in indices]

        values_nan = np.where(mask > 0, values, np.nan).astype(np.float32)

        return Batch(
            times=torch.from_numpy(self.times),
            values=torch.from_numpy(values_nan),
            mask=torch.from_numpy(mask),
            delta_t=torch.from_numpy(delta_t),
            static=torch.from_numpy(static),
            labels=torch.from_numpy(labels),
            record_ids=record_ids,
        )
