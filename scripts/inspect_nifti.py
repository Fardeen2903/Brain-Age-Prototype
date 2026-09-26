from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    img = nib.as_closest_canonical(nib.load(args.path))
    arr = img.get_fdata(dtype=np.float32)
    print(f"shape={arr.shape}")
    print(f"voxel sizes={img.header.get_zooms()[:3]}")
    print(f"min={np.nanmin(arr):.3f} max={np.nanmax(arr):.3f} mean={np.nanmean(arr):.3f}")

    cx, cy, cz = [s // 2 for s in arr.shape]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(arr[cx, :, :].T, cmap="gray", origin="lower")
    axes[0].set_title("Sagittal-ish")
    axes[1].imshow(arr[:, cy, :].T, cmap="gray", origin="lower")
    axes[1].set_title("Coronal-ish")
    axes[2].imshow(arr[:, :, cz].T, cmap="gray", origin="lower")
    axes[2].set_title("Axial-ish")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
