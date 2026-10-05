"""Inference for the final GRU mortality model."""
from __future__ import annotations

import numpy as np
import torch

from src.data.preprocessing import Physionet2012Dataset, fill_forward
from src.model_bundle import load_bundle


@torch.no_grad()
def predict_records(
    records,
    bundle_path,
    *,
    device="cpu",
    batch_size=128,
):
    artifact, model, normalizer = load_bundle(bundle_path, device=device)

    dataset = Physionet2012Dataset(
        records,
        vocab=artifact["vocab"],
        static_names=artifact["static_names"],
        normalizer=normalizer,
    )

    scores = []
    record_ids = []

    for start in range(0, len(dataset), batch_size):
        indices = list(
            range(start, min(start + batch_size, len(dataset)))
        )
        batch = dataset.collate(indices).to(device)

        filled = torch.from_numpy(
            fill_forward(
                batch.values.detach().cpu().numpy(),
                batch.mask.detach().cpu().numpy(),
            ).astype(np.float32)
        ).to(device)

        logits = model(batch, filled)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        record_ids.extend(batch.record_ids)

    score = (
        np.concatenate(scores)
        if scores
        else np.empty(0, dtype=np.float32)
    )
    threshold = float(artifact["threshold"])

    return {
        "record_id": np.asarray(record_ids),
        "score": score,
        "prediction": (score >= threshold).astype(np.int64),
    }
