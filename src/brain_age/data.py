from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

REQUIRED_COLUMNS = {"subject_id", "scan_id", "file_path", "age", "diagnosis"}


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    base = csv_path.parent
    df = df.copy()
    df["file_path"] = df["file_path"].map(
        lambda p: str((base / p).resolve()) if not Path(str(p)).is_absolute() else str(p)
    )
    return df


def load_nifti_tensor(path: str | Path, target_shape: tuple[int, int, int]) -> torch.Tensor:
    img = nib.as_closest_canonical(nib.load(str(path)))
    volume = img.get_fdata(dtype=np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

    nonzero = volume != 0
    if np.any(nonzero):
        values = volume[nonzero]
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-6:
            std = 1.0
        volume[nonzero] = (values - mean) / std

    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
    return tensor.squeeze(0)  # [C, D, H, W]


class BrainAgeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, target_shape: tuple[int, int, int]):
        self.df = dataframe.reset_index(drop=True).copy()
        self.target_shape = target_shape

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        image = load_nifti_tensor(row.file_path, self.target_shape)
        return {
            "image": image,
            "age": torch.tensor(float(row.age), dtype=torch.float32),
            "subject_id": str(row.subject_id),
            "scan_id": str(row.scan_id),
            "diagnosis": str(row.diagnosis),
        }
