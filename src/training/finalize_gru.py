"""Train the selected GRU on all Set A records and export a model bundle."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.physionet import load_split
from src.data.preprocessing import (
    Physionet2012Dataset,
    build_vocab,
    fill_forward,
)
from src.model_bundle import save_bundle
from src.models.gru import GRUMortalityModel


SEED = 42
EPOCHS = 11
BATCH_SIZE = 128
LR = 1e-3
GRAD_CLIP = 5.0
THRESHOLD = 0.2571
HIDDEN = 64


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def iter_batches(dataset, epoch):
    indices = np.arange(len(dataset))
    rng = np.random.RandomState(SEED + int(epoch))
    rng.shuffle(indices)

    for start in range(0, len(indices), BATCH_SIZE):
        yield dataset.collate(
            indices[start:start + BATCH_SIZE].tolist()
        )


def train(data_dir, output_path, device):
    records = load_split(Path(data_dir), split="set-a")
    if not records or any(record.label is None for record in records):
        raise RuntimeError("Set A must contain labels.")

    vocab = build_vocab(records)
    dataset = Physionet2012Dataset(records, vocab)
    normalizer = dataset.fit_normalizer()

    set_seed()

    model = GRUMortalityModel(
        input_dim=len(vocab),
        static_dim=len(dataset.static_names),
        hidden=HIDDEN,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for batch in iter_batches(dataset, epoch):
            batch = batch.to(device)

            filled = torch.from_numpy(
                fill_forward(
                    batch.values.detach().cpu().numpy(),
                    batch.mask.detach().cpu().numpy(),
                ).astype(np.float32)
            ).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch, filled)
            loss = loss_fn(logits, batch.labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )
            optimizer.step()

            bs = len(batch.labels)
            total_loss += float(loss.detach()) * bs
            n += bs

        print(
            f"epoch {epoch:02d}/{EPOCHS:02d} "
            f"loss={total_loss / n:.4f}"
        )

    save_bundle(
        output_path,
        model=model,
        vocab=vocab,
        static_names=dataset.static_names,
        normalizer=normalizer,
        threshold=THRESHOLD,
        hidden=HIDDEN,
    )

    print(f"saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--out",
        default="artifacts/gru_physionet2012.pt",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_path=args.out,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
