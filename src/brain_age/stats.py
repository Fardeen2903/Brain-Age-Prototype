from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled_var = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((x.mean() - y.mean()) / np.sqrt(pooled_var))


def summarize_groups(df: pd.DataFrame, bag_col: str = "bag_corrected") -> dict:
    groups = {}
    for diagnosis, group in df.groupby("diagnosis"):
        values = group[bag_col].dropna().to_numpy(float)
        n = len(values)
        mean = float(np.mean(values)) if n else float("nan")
        sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
        sem = sd / np.sqrt(n) if n > 1 else float("nan")
        ci = [mean - 1.96 * sem, mean + 1.96 * sem] if n > 1 else [float("nan")] * 2
        groups[str(diagnosis)] = {"n": n, "mean_bag": mean, "sd": sd, "ci95": ci}

    ordered = [g for g in ["CN", "MCI", "AD"] if g in set(df["diagnosis"])]
    arrays = [df.loc[df["diagnosis"] == g, bag_col].dropna().to_numpy(float) for g in ordered]
    anova = {}
    if len(arrays) >= 2 and all(len(a) > 1 for a in arrays):
        f_stat, p_value = stats.f_oneway(*arrays)
        anova = {"groups": ordered, "F": float(f_stat), "p": float(p_value)}

    effects = {}
    for a, b in combinations(ordered, 2):
        effects[f"{a}_vs_{b}"] = cohens_d(
            df.loc[df["diagnosis"] == a, bag_col].to_numpy(float),
            df.loc[df["diagnosis"] == b, bag_col].to_numpy(float),
        )

    return {"groups": groups, "anova": anova, "cohens_d": effects}
