from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from brain_age.config import load_config
from brain_age.data import BrainAgeDataset, load_metadata
from brain_age.models import build_model
from brain_age.split import attach_split_labels, split_cn_subjects
from brain_age.utils import ensure_dir, resolve_device, set_seed


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["age"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss = loss_fn(pred, y)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * len(y)
        total_n += len(y)

    return total_loss / max(total_n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    output_dir = ensure_dir(cfg["output_dir"])
    device = resolve_device(cfg["training"]["device"])

    df = load_metadata(cfg["data"]["csv"])
    train_df, val_df, test_df = split_cn_subjects(
        df,
        train_fraction=float(cfg["data"]["train_fraction"]),
        val_fraction=float(cfg["data"]["val_fraction"]),
        test_fraction=float(cfg["data"]["test_fraction"]),
        seed=int(cfg["seed"]),
    )
    split_df = attach_split_labels(df, train_df, val_df, test_df)
    split_df.to_csv(output_dir / "splits.csv", index=False)

    shape = tuple(int(x) for x in cfg["data"]["target_shape"])
    train_ds = BrainAgeDataset(train_df, shape)
    val_ds = BrainAgeDataset(val_df, shape)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
    )

    model = build_model(cfg["model"]["name"]).to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    best_val = float("inf")
    history = []
    print(f"Device: {device}")
    print(f"CN scans: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_mae = run_epoch(model, train_loader, loss_fn, device, optimizer)
        with torch.no_grad():
            val_mae = run_epoch(model, val_loader, loss_fn, device)

        history.append({"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae})
        print(f"Epoch {epoch:02d} | train MAE={train_mae:.3f} | val MAE={val_mae:.3f}")

        if val_mae < best_val:
            best_val = val_mae
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": cfg["model"]["name"],
                    "target_shape": shape,
                },
                output_dir / "best_model.pt",
            )

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    print(f"Saved best model to {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
