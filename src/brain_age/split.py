from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_cn_subjects(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-8:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0")

    cn = df[df["diagnosis"].str.upper() == "CN"].copy()
    if cn.empty:
        raise ValueError("No CN rows found. The normative model must be trained on CN subjects.")

    first = GroupShuffleSplit(
        n_splits=1,
        train_size=train_fraction,
        random_state=seed,
    )
    train_idx, remainder_idx = next(first.split(cn, groups=cn["subject_id"]))
    train = cn.iloc[train_idx].copy()
    remainder = cn.iloc[remainder_idx].copy()

    relative_val = val_fraction / (val_fraction + test_fraction)
    second = GroupShuffleSplit(
        n_splits=1,
        train_size=relative_val,
        random_state=seed + 1,
    )
    val_idx, test_idx = next(second.split(remainder, groups=remainder["subject_id"]))
    val = remainder.iloc[val_idx].copy()
    test = remainder.iloc[test_idx].copy()

    assert_subject_disjoint(train, val, test)
    return train, val, test


def assert_subject_disjoint(*frames: pd.DataFrame) -> None:
    sets = [set(frame["subject_id"].astype(str)) for frame in frames]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise AssertionError(f"Subject leakage detected: {sorted(overlap)[:5]}")


def attach_split_labels(
    full_df: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    out = full_df.copy()
    out["split"] = "clinical"
    key = out["scan_id"].astype(str)
    out.loc[key.isin(train["scan_id"].astype(str)), "split"] = "train"
    out.loc[key.isin(val["scan_id"].astype(str)), "split"] = "val"
    out.loc[key.isin(test["scan_id"].astype(str)), "split"] = "test"
    return out
