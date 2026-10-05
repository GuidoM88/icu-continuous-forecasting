"""Serialization helpers for the final GRU model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.preprocessing import Normalizer
from src.models.gru import GRUMortalityModel


def save_bundle(
    path,
    *,
    model,
    vocab,
    static_names,
    normalizer,
    threshold,
    hidden=64,
):
    artifact = {
        "model_name": "gru",
        "model_kwargs": {"hidden": int(hidden)},
        "vocab": list(vocab),
        "static_names": list(static_names),
        "normalizer": {
            "mean": np.asarray(normalizer.mean, dtype=np.float32),
            "std": np.asarray(normalizer.std, dtype=np.float32),
            "static_mean": np.asarray(normalizer.static_mean, dtype=np.float32),
            "static_std": np.asarray(normalizer.static_std, dtype=np.float32),
        },
        "threshold": float(threshold),
        "state_dict": model.state_dict(),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def load_bundle(path, device="cpu"):
    artifact = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )

    norm = artifact["normalizer"]
    normalizer = Normalizer(
        mean=np.asarray(norm["mean"], dtype=np.float32),
        std=np.asarray(norm["std"], dtype=np.float32),
        static_mean=np.asarray(norm["static_mean"], dtype=np.float32),
        static_std=np.asarray(norm["static_std"], dtype=np.float32),
    )

    model = GRUMortalityModel(
        input_dim=len(artifact["vocab"]),
        static_dim=len(artifact["static_names"]),
        **artifact["model_kwargs"],
    )
    model.load_state_dict(artifact["state_dict"])
    model.to(device)
    model.eval()

    return artifact, model, normalizer
