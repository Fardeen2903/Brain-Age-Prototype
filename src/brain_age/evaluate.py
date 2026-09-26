from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

from brain_age.bias_correction import AgeBiasCorrector
from brain_age.config import load_config
from brain_age.data import BrainAgeDataset
from brain_age.models import build_model
from brain_age.stats import summarize_groups
from brain_age.utils import resolve_device, save_json


def predict_frame(model, frame, target_shape, device, batch_size, num_workers):
    ds = BrainAgeDataset(frame, target_shape)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["image"].to(device)).cpu().numpy()
            ages = batch["age"].numpy()
            for i in range(len(pred)):
                rows.append(
                    {
                        "subject_id": batch["subject_id"][i],
                        "scan_id": batch["scan_id"][i],
                        "diagnosis": batch["diagnosis"][i],
                        "age": float(ages[i]),
                        "predicted_age": float(pred[i]),
                    }
                )
    return pd.DataFrame(rows)


def save_plots(df: pd.DataFrame, output_dir: Path) -> None:
    eval_df = df[df["analysis_set"] == "evaluation"].copy()

    fig = plt.figure(figsize=(6, 6))
    for diagnosis, group in eval_df.groupby("diagnosis"):
        plt.scatter(group["age"], group["predicted_age"], label=diagnosis, alpha=0.7)
    lo = min(eval_df["age"].min(), eval_df["predicted_age"].min())
    hi = max(eval_df["age"].max(), eval_df["predicted_age"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Chronological age")
    plt.ylabel("Predicted age")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "predicted_vs_age.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(eval_df["age"], eval_df["bag_raw"], alpha=0.6, label="Raw BAG")
    plt.scatter(eval_df["age"], eval_df["bag_corrected"], alpha=0.6, label="Corrected BAG")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Chronological age")
    plt.ylabel("Brain age gap (years)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "bag_bias_correction.png", dpi=160)
    plt.close(fig)

    groups = [g for g in ["CN", "MCI", "AD"] if g in set(eval_df["diagnosis"])]
    if groups:
        values = [eval_df.loc[eval_df["diagnosis"] == g, "bag_corrected"] for g in groups]
        fig = plt.figure(figsize=(6, 5))
        plt.boxplot(values, tick_labels=groups)
        plt.axhline(0, linestyle="--")
        plt.ylabel("Corrected BAG (years)")
        plt.tight_layout()
        fig.savefig(output_dir / "bag_by_group.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    device = resolve_device(cfg["training"]["device"])
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)

    split_df = pd.read_csv(output_dir / "splits.csv")
    shape = tuple(int(x) for x in checkpoint["target_shape"])
    model = build_model(checkpoint["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    val_df = split_df[split_df["split"] == "val"].copy()
    eval_df = split_df[split_df["split"].isin(["test", "clinical"])].copy()

    val_pred = predict_frame(
        model,
        val_df,
        shape,
        device,
        int(cfg["training"]["batch_size"]),
        int(cfg["data"]["num_workers"]),
    )
    eval_pred = predict_frame(
        model,
        eval_df,
        shape,
        device,
        int(cfg["training"]["batch_size"]),
        int(cfg["data"]["num_workers"]),
    )

    corrector = AgeBiasCorrector.fit(val_pred["age"].to_numpy(), val_pred["predicted_age"].to_numpy())
    save_json(corrector.to_dict(), output_dir / "age_bias_correction.json")

    val_pred["analysis_set"] = "calibration"
    eval_pred["analysis_set"] = "evaluation"
    pred = pd.concat([val_pred, eval_pred], ignore_index=True)
    pred["bag_raw"] = pred["predicted_age"] - pred["age"]
    pred["corrected_predicted_age"] = corrector.correct_predicted_age(pred["predicted_age"].to_numpy())
    pred["bag_corrected"] = pred["corrected_predicted_age"] - pred["age"]
    pred.to_csv(output_dir / "predictions.csv", index=False)

    metric_rows = []
    only_eval = pred[pred["analysis_set"] == "evaluation"]
    for diagnosis, group in only_eval.groupby("diagnosis"):
        mae = mean_absolute_error(group["age"], group["predicted_age"])
        r = float("nan")
        if len(group) > 1 and group["age"].nunique() > 1 and group["predicted_age"].nunique() > 1:
            r = float(pearsonr(group["age"], group["predicted_age"]).statistic)
        metric_rows.append({"diagnosis": diagnosis, "n": len(group), "mae": mae, "pearson_r": r})
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics_by_group.csv", index=False)

    stats_raw = summarize_groups(only_eval, bag_col="bag_raw")
    stats_corrected = summarize_groups(only_eval, bag_col="bag_corrected")
    save_json({"raw": stats_raw, "corrected": stats_corrected}, output_dir / "bag_statistics.json")
    save_plots(pred, output_dir)

    raw_corr = np.corrcoef(only_eval["age"], only_eval["bag_raw"])[0, 1]
    corrected_corr = np.corrcoef(only_eval["age"], only_eval["bag_corrected"])[0, 1]
    print(f"Raw corr(age, BAG):       {raw_corr:.3f}")
    print(f"Corrected corr(age, BAG): {corrected_corr:.3f}")
    print(f"Bias correction: predicted = {corrector.intercept:.3f} + {corrector.slope:.3f} * age")
    print(f"Saved evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
