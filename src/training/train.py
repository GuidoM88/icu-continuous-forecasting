"""
Set-A development utility for comparing the research models.

Model selection is restricted to a stratified Set-A train/validation split.
Sets B and C are intentionally absent from this script. Final all-A training
and artifact export for the selected GRU are handled by
`src.training.finalize_gru`.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.physionet import load_split
from src.data.preprocessing import Physionet2012Dataset, build_vocab, fill_forward
from src.data.synthetic import make_synthetic_records
from src.models.baselines import GRUD, GRUBaseline, PersistenceBaseline
from src.models.latent_ode import LatentODE, kl_standard_normal, masked_mse
from src.models.neural_cde import NeuralCDE
from src.training.metrics import (
    challenge_metrics,
    classification_metrics,
    select_event1_threshold,
)

ALL_MODELS = ["persistence", "gru", "grud", "latent_ode", "neural_cde"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_split(records, val_frac: float = 0.20, seed: int = 42):
    rng = random.Random(seed)
    by_label = {0: [], 1: []}

    for i, record in enumerate(records):
        if record.label is None:
            continue
        by_label[int(record.label)].append(i)

    train_idx, val_idx = [], []

    for indices in by_label.values():
        rng.shuffle(indices)
        n_val = int(round(len(indices) * val_frac))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def iterate_batches(
    dataset,
    indices,
    batch_size,
    *,
    shuffle=False,
    seed=42,
):
    ids = np.asarray(indices, dtype=int).copy()

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(ids)

    for start in range(0, len(ids), batch_size):
        yield dataset.collate(ids[start : start + batch_size].tolist())


def build_model(name: str, input_dim: int, static_dim: int, device):
    if name == "persistence":
        model = PersistenceBaseline(input_dim, static_dim)
    elif name == "gru":
        model = GRUBaseline(input_dim, static_dim)
    elif name == "grud":
        model = GRUD(input_dim, static_dim)
    elif name == "latent_ode":
        model = LatentODE(
            input_dim,
            static_dim,
            latent_dim=16,
            hidden=64,
            solver="dopri5",
        )
    elif name == "neural_cde":
        model = NeuralCDE(
            input_dim,
            static_dim,
            hidden_channels=32,
            hidden=64,
            interpolation="cubic",
            solver="rk4",
            step_size=1.0,
        )
    else:
        raise ValueError(f"Unknown model {name!r}")

    return model.to(device)


def filled_from_batch(batch):
    values = batch.values.detach().cpu().numpy()
    mask = batch.mask.detach().cpu().numpy()
    filled = fill_forward(values, mask).astype(np.float32)
    return torch.from_numpy(filled).to(batch.values.device)


def forward_logits(name, model, batch, *, training: bool):
    if name == "persistence":
        return model(batch), None

    if name == "gru":
        return model(batch, filled_from_batch(batch)), None

    if name == "grud":
        x_mean = torch.zeros(
            batch.values.shape[-1],
            dtype=batch.values.dtype,
            device=batch.values.device,
        )
        return model(batch, x_mean), None

    if name == "latent_ode":
        logits, recon, mean, logvar = model(
            batch,
            filled_from_batch(batch),
            sample=training,
        )
        return logits, (recon, mean, logvar)

    if name == "neural_cde":
        return model(batch), None

    raise ValueError(name)


def train_epoch(
    name,
    model,
    dataset,
    indices,
    optimizer,
    batch_size,
    device,
    epoch,
    seed,
):
    model.train()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_n = 0

    for batch in iterate_batches(
        dataset,
        indices,
        batch_size,
        shuffle=True,
        seed=seed + epoch,
    ):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits, aux = forward_logits(name, model, batch, training=True)
        loss = bce(logits, batch.labels.float())

        if name == "latent_ode":
            recon, mean, logvar = aux
            loss = (
                loss
                + 0.1 * masked_mse(recon, batch.values, batch.mask).mean()
                + 1e-3 * kl_standard_normal(mean, logvar).mean()
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        n = len(batch.labels)
        total_loss += float(loss.detach()) * n
        total_n += n

    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(
    name,
    model,
    dataset,
    indices,
    batch_size,
    device,
):
    model.eval()
    labels, scores = [], []

    for batch in iterate_batches(
        dataset,
        indices,
        batch_size,
        shuffle=False,
    ):
        batch = batch.to(device)
        logits, _ = forward_logits(name, model, batch, training=False)
        labels.append(batch.labels.cpu().numpy())
        scores.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(labels), np.concatenate(scores)


def train_one_model(
    name,
    dataset,
    train_idx,
    val_idx,
    *,
    input_dim,
    static_dim,
    device,
    max_epochs,
    patience,
    batch_size,
    lr,
    seed,
):
    set_seed(seed)
    model = build_model(name, input_dim, static_dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auprc = -np.inf
    best_state = None
    best_epoch = None
    stale = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()

        train_loss = train_epoch(
            name,
            model,
            dataset,
            train_idx,
            optimizer,
            batch_size,
            device,
            epoch,
            seed,
        )

        y_val, risk_val = predict(
            name,
            model,
            dataset,
            val_idx,
            batch_size,
            device,
        )
        metrics = classification_metrics(y_val, risk_val)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auroc": metrics["auroc"],
                "val_auprc": metrics["auprc"],
                "seconds": time.perf_counter() - t0,
            }
        )

        improved = (
            np.isfinite(metrics["auprc"])
            and metrics["auprc"] > best_auprc
        )

        if improved:
            best_auprc = metrics["auprc"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        print(
            f"[{name}] {epoch:02d} "
            f"loss={train_loss:.4f} "
            f"val_AUROC={metrics['auroc']:.4f} "
            f"val_AUPRC={metrics['auprc']:.4f}"
        )

        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"No valid validation checkpoint for {name}")

    model.load_state_dict(best_state)

    y_val, risk_val = predict(
        name,
        model,
        dataset,
        val_idx,
        batch_size,
        device,
    )
    threshold = select_event1_threshold(y_val, risk_val)
    val_metrics = challenge_metrics(y_val, risk_val, threshold)

    return {
        "selected_epoch": int(best_epoch),
        "threshold": float(threshold),
        "validation": val_metrics,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-n", type=int, default=600)
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--out",
        default="results/development.json",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    if args.synthetic:
        records = make_synthetic_records(
            n=args.synthetic_n,
            seed=args.seed,
        )
    else:
        records = load_split(
            Path(args.data_dir),
            split="set-a",
        )

    records = [record for record in records if record.label is not None]

    train_idx, val_idx = stratified_split(
        records,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    train_records = [records[i] for i in train_idx]
    vocab = build_vocab(train_records)

    train_ds = Physionet2012Dataset(train_records, vocab)
    normalizer = train_ds.fit_normalizer()

    dataset = Physionet2012Dataset(
        records,
        vocab,
        normalizer=normalizer,
    )

    print(
        f"records={len(records)} "
        f"train={len(train_idx)} "
        f"val={len(val_idx)} "
        f"vocab={len(vocab)} "
        f"device={device}"
    )

    results = {}

    for name in args.models:
        print(f"\n=== {name} ===")
        results[name] = train_one_model(
            name,
            dataset,
            train_idx,
            val_idx,
            input_dim=len(vocab),
            static_dim=len(dataset.static_names),
            device=device,
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n=== Set-A validation summary ===")
    print(
        f"{'model':<14}"
        f"{'epoch':>8}"
        f"{'AUROC':>10}"
        f"{'AUPRC':>10}"
        f"{'Event1':>10}"
        f"{'threshold':>12}"
    )

    for name, result in results.items():
        metrics = result["validation"]
        print(
            f"{name:<14}"
            f"{result['selected_epoch']:>8}"
            f"{metrics['auroc']:>10.4f}"
            f"{metrics['auprc']:>10.4f}"
            f"{metrics['event1']:>10.4f}"
            f"{result['threshold']:>12.4f}"
        )

    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
