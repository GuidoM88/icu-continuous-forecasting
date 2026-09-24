"""
Trains and compares every model in this repo on in-hospital-mortality
classification, and prints/saves a results table.

Usage:
    # instant sanity run on fake data, no download needed:
    python -m src.training.train --synthetic --epochs 3

    # real PhysioNet-2012 data (after running data/download_physionet2012.sh):
    python -m src.training.train --data-dir data/raw --epochs 30

    # just one or two models, e.g. while iterating on Neural CDE:
    python -m src.training.train --synthetic --models persistence neural_cde
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.physionet import load_split
from src.data.preprocessing import Physionet2012Dataset, build_vocab
from src.data.synthetic import make_synthetic_records
from src.models.baselines import GRUD, GRUBaseline, PersistenceBaseline
from src.models.latent_ode import LatentODE, kl_standard_normal, masked_mse
from src.models.neural_cde import NeuralCDE
from src.training.metrics import classification_metrics

ALL_MODELS = ["persistence", "gru", "grud", "latent_ode", "neural_cde"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_split(records, val_frac=0.1, test_frac=0.1, seed=0):
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for i, r in enumerate(records):
        if r.label is not None:
            by_label[r.label].append(i)
    train_idx, val_idx, test_idx = [], [], []
    for label, idxs in by_label.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_val, n_test = int(n * val_frac), int(n * test_frac)
        val_idx += idxs[:n_val]
        test_idx += idxs[n_val : n_val + n_test]
        train_idx += idxs[n_val + n_test :]
    rng.shuffle(train_idx)
    return train_idx, val_idx, test_idx


def iterate_batches(dataset, indices, batch_size, shuffle=True, seed=0):
    idx = list(indices)
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = idx[i : i + batch_size]
        if len(chunk) < 2:  # BatchNorm-free models don't need this, but AUROC does
            continue
        yield dataset.collate(chunk)


def build_model(name: str, input_dim: int, static_dim: int, device):
    if name == "persistence":
        return PersistenceBaseline(input_dim, static_dim).to(device)
    if name == "gru":
        return GRUBaseline(input_dim, static_dim).to(device)
    if name == "grud":
        return GRUD(input_dim, static_dim).to(device)
    if name == "latent_ode":
        return LatentODE(input_dim, static_dim).to(device)
    if name == "neural_cde":
        return NeuralCDE(input_dim, static_dim).to(device)
    raise ValueError(f"Unknown model {name!r}, choose from {ALL_MODELS}")


def run_epoch(model_name, model, dataset, indices, batch_size, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    bce = nn.BCEWithLogitsLoss()

    all_labels, all_scores, losses = [], [], []
    for batch in iterate_batches(dataset, indices, batch_size, shuffle=train_mode):
        batch = batch.to(device)
        from src.data.preprocessing import fill_forward

        filled = None
        if model_name in ("gru", "latent_ode"):
            filled = torch.from_numpy(
                fill_forward(
                    torch.nan_to_num(batch.values, nan=float("nan")).cpu().numpy(),
                    batch.mask.cpu().numpy(),
                )
            ).to(device)

        if model_name == "latent_ode":
            logits, recon, mean, logvar = model(batch, filled)
            recon_loss = masked_mse(recon, batch.values, batch.mask).mean()
            kl = kl_standard_normal(mean, logvar).mean()
            cls_loss = bce(logits, batch.labels)
            loss = cls_loss + 0.1 * recon_loss + 1e-3 * kl
        elif model_name in ("gru", "grud"):
            if model_name == "grud":
                x_mean = torch.zeros(batch.values.shape[-1])  # values are normalised -> mean 0
                logits = model(batch, x_mean)
            else:
                logits = model(batch, filled)
            loss = bce(logits, batch.labels)
        else:  # persistence, neural_cde
            logits = model(batch)
            loss = bce(logits, batch.labels)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        losses.append(loss.item())
        all_labels.append(batch.labels.detach().cpu().numpy())
        all_scores.append(torch.sigmoid(logits).detach().cpu().numpy())

    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    y_score = np.concatenate(all_scores) if all_scores else np.array([])
    metrics = classification_metrics(y_true, y_score)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def train_one_model(name, dataset_train, dataset_val, dataset_test, train_idx, val_idx, test_idx,
                     input_dim, static_dim, device, epochs, batch_size, lr):
    model = build_model(name, input_dim, static_dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auroc, best_state = -1.0, None
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        train_m = run_epoch(name, model, dataset_train, train_idx, batch_size, device, optimizer)
        with torch.no_grad():
            val_m = run_epoch(name, model, dataset_val, val_idx, batch_size, device, optimizer=None)
        history.append({"epoch": epoch, "train": train_m, "val": val_m, "sec": time.time() - t0})
        print(
            f"  [{name}] epoch {epoch:02d}  train_loss={train_m['loss']:.4f}  "
            f"val_auroc={val_m['auroc']:.4f}  val_auprc={val_m['auprc']:.4f}  "
            f"({time.time() - t0:.1f}s)"
        )
        if val_m["auroc"] == val_m["auroc"] and val_m["auroc"] > best_val_auroc:  # not NaN
            best_val_auroc = val_m["auroc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        test_m = run_epoch(name, model, dataset_test, test_idx, batch_size, device, optimizer=None)
    return model, test_m, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--synthetic", action="store_true", help="use fake data, no download needed")
    parser.add_argument("--synthetic-n", type=int, default=600)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/results.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    if args.synthetic:
        print(f"Generating {args.synthetic_n} synthetic records (no PhysioNet download needed)...")
        records = make_synthetic_records(n=args.synthetic_n, seed=args.seed)
    else:
        print(f"Loading real PhysioNet-2012 records from {args.data_dir} ...")
        records = load_split(Path(args.data_dir), split="set-a")

    n_before = len(records)
    records = [r for r in records if r.label is not None]
    if len(records) < n_before:
        print(f"dropped {n_before - len(records)} records with no known outcome")

    train_idx, val_idx, test_idx = stratified_split(records, seed=args.seed)
    print(f"records: {len(records)}  train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    vocab = build_vocab([records[i] for i in train_idx])
    print(f"variable vocabulary size: {len(vocab)}")

    # Fit the normalizer on the TRAIN split only (never on val/test), then reuse
    # those statistics for every split -- standard practice to avoid leakage.
    train_records = [records[i] for i in train_idx]
    normalizer = Physionet2012Dataset(train_records, vocab).fit_normalizer()
    ds = Physionet2012Dataset(records, vocab, normalizer=normalizer)
    input_dim, static_dim = len(vocab), len(ds.static_names)

    results = {}
    for name in args.models:
        print(f"\n=== training {name} ===")
        _, test_m, history = train_one_model(
            name, ds, ds, ds, train_idx, val_idx, test_idx,
            input_dim, static_dim, device, args.epochs, args.batch_size, args.lr,
        )
        print(f"  -> TEST  auroc={test_m['auroc']:.4f}  auprc={test_m['auprc']:.4f}")
        results[name] = {"test": test_m, "history": history}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== summary (test set) ===")
    print(f"{'model':<14}{'AUROC':>10}{'AUPRC':>10}")
    for name in args.models:
        m = results[name]["test"]
        print(f"{name:<14}{m['auroc']:>10.4f}{m['auprc']:>10.4f}")
    print(f"\nsaved full results to {out_path}")


if __name__ == "__main__":
    main()
