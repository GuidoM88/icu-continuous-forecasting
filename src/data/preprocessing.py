"""Hourly preprocessing used by the final GRU model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import torch

from .physionet import GENERAL_DESCRIPTORS, PatientRecord


def build_vocab(records: Sequence[PatientRecord]) -> List[str]:
    vocab = set()
    for record in records:
        vocab.update(v.strip() for v in record.values if v and v.strip())
    return sorted(vocab)


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray

    @staticmethod
    def fit(values: np.ndarray, mask: np.ndarray, static: np.ndarray) -> "Normalizer":
        D = values.shape[-1]
        mean = np.zeros(D, dtype=np.float32)
        std = np.ones(D, dtype=np.float32)

        for d in range(D):
            observed = values[..., d][mask[..., d] > 0]
            if observed.size:
                mean[d] = observed.mean()
                s = observed.std()
                std[d] = s if s > 1e-6 else 1.0

        static_mean = np.nanmean(static, axis=0)
        static_std = np.nanstd(static, axis=0)
        static_mean = np.nan_to_num(static_mean, nan=0.0)
        static_std = np.where(
            np.isfinite(static_std) & (static_std > 1e-6),
            static_std,
            1.0,
        )

        return Normalizer(
            mean=mean,
            std=std,
            static_mean=static_mean.astype(np.float32),
            static_std=static_std.astype(np.float32),
        )

    def apply(
        self,
        values: np.ndarray,
        static: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        values = (values - self.mean) / self.std
        static = (static - self.static_mean) / self.static_std
        static = np.nan_to_num(static, nan=0.0)
        return values, static


@dataclass
class Batch:
    times: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor
    delta_t: torch.Tensor
    static: torch.Tensor
    labels: torch.Tensor
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


def compute_delta_t(mask: np.ndarray, bin_size_hours: float = 1.0) -> np.ndarray:
    B, T, D = mask.shape
    delta = np.zeros((B, T, D), dtype=np.float32)

    for t in range(1, T):
        observed_prev = mask[:, t - 1] > 0
        delta[:, t] = np.where(
            observed_prev,
            bin_size_hours,
            bin_size_hours + delta[:, t - 1],
        )

    return delta


def fill_forward(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    B, T, D = values.shape
    out = values.copy()
    last = np.zeros((B, D), dtype=np.float32)
    seen = np.zeros((B, D), dtype=bool)

    for t in range(T):
        observed = mask[:, t] > 0
        current = np.where(
            observed,
            np.nan_to_num(out[:, t], nan=0.0),
            last,
        )
        out[:, t] = np.where(seen | observed, current, 0.0)
        last = current
        seen |= observed

    return out


class Physionet2012Dataset:
    def __init__(
        self,
        records: Sequence[PatientRecord],
        vocab: Sequence[str],
        static_names: Sequence[str] = tuple(GENERAL_DESCRIPTORS),
        max_hours: float = 48.0,
        bin_size_hours: float = 1.0,
        normalizer: "Normalizer | None" = None,
    ):
        self.records = list(records)
        self.vocab = list(vocab)
        self.static_names = list(static_names)
        self.var_index = {v: i for i, v in enumerate(self.vocab)}
        self.max_hours = float(max_hours)
        self.bin_size_hours = float(bin_size_hours)
        self.n_bins = int(self.max_hours // self.bin_size_hours) + 1
        self.times = np.arange(self.n_bins, dtype=np.float32) * self.bin_size_hours
        self.normalizer = normalizer

    def __len__(self):
        return len(self.records)

    def fit_normalizer(self) -> Normalizer:
        raw = [self._bin_one(i, normalize=False) for i in range(len(self))]
        values = np.stack([x[0] for x in raw])
        mask = np.stack([x[1] for x in raw])
        static = np.stack([x[2] for x in raw])
        self.normalizer = Normalizer.fit(values, mask, static)
        return self.normalizer

    def _bin_one(self, idx: int, normalize: bool = True):
        rec = self.records[idx]
        D = len(self.vocab)

        values = np.zeros((self.n_bins, D), dtype=np.float32)
        counts = np.zeros((self.n_bins, D), dtype=np.float32)

        for var, points in rec.values.items():
            j = self.var_index.get(var)
            if j is None:
                continue

            for t, value in points:
                if not (0.0 <= t <= self.max_hours):
                    continue

                b = min(
                    int(round(t / self.bin_size_hours)),
                    self.n_bins - 1,
                )
                values[b, j] += value
                counts[b, j] += 1.0

        mask = (counts > 0).astype(np.float32)
        observed = counts > 0
        values[observed] /= counts[observed]

        static = np.asarray(
            [rec.static.get(name, np.nan) for name in self.static_names],
            dtype=np.float32,
        )

        if normalize and self.normalizer is not None:
            values, static = self.normalizer.apply(values, static)

        label = np.nan if rec.label is None else float(rec.label)
        return values, mask, static, label

    def collate(self, indices: Sequence[int]) -> Batch:
        if self.normalizer is None:
            raise RuntimeError("A fitted normalizer is required.")

        raw = [self._bin_one(i) for i in indices]

        values = np.stack([x[0] for x in raw])
        mask = np.stack([x[1] for x in raw])
        static = np.stack([x[2] for x in raw])
        labels = np.asarray([x[3] for x in raw], dtype=np.float32)
        delta_t = compute_delta_t(mask, self.bin_size_hours)
        record_ids = [self.records[i].record_id for i in indices]

        values = np.where(mask > 0, values, np.nan).astype(np.float32)

        return Batch(
            times=torch.from_numpy(self.times),
            values=torch.from_numpy(values),
            mask=torch.from_numpy(mask),
            delta_t=torch.from_numpy(delta_t),
            static=torch.from_numpy(static),
            labels=torch.from_numpy(labels),
            record_ids=record_ids,
        )
