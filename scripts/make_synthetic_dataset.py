from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def synthetic_brain(age: float, diagnosis: str, shape=(48, 48, 48), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, shape[0]),
        np.linspace(-1, 1, shape[1]),
        np.linspace(-1, 1, shape[2]),
        indexing="ij",
    )

    brain = ((x / 0.78) ** 2 + (y / 0.92) ** 2 + (z / 0.82) ** 2) <= 1.0
    volume = np.zeros(shape, dtype=np.float32)
    volume[brain] = 0.9

    disease_offset = {"CN": 0.0, "MCI": 4.0, "AD": 8.0}[diagnosis]
    apparent_age = age + disease_offset
    ventricle_radius = 0.09 + 0.0025 * (apparent_age - 55.0)
    ventricle_radius = float(np.clip(ventricle_radius, 0.08, 0.19))

    left = ((x + 0.13) / ventricle_radius) ** 2 + (y / (ventricle_radius * 1.35)) ** 2 + (z / (ventricle_radius * 1.6)) ** 2 <= 1
    right = ((x - 0.13) / ventricle_radius) ** 2 + (y / (ventricle_radius * 1.35)) ** 2 + (z / (ventricle_radius * 1.6)) ** 2 <= 1
    volume[left | right] = 0.15

    cortical_shrink = 1.0 - 0.0015 * (apparent_age - 55.0)
    peripheral = ((x / (0.78 * cortical_shrink)) ** 2 + (y / (0.92 * cortical_shrink)) ** 2 + (z / (0.82 * cortical_shrink)) ** 2) <= 1.0
    volume[brain & ~peripheral] *= 0.6

    volume += rng.normal(0, 0.035, size=shape).astype(np.float32) * brain
    volume[~brain] = 0.0
    return volume


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--cn-subjects", type=int, default=60)
    parser.add_argument("--mci-subjects", type=int, default=12)
    parser.add_argument("--ad-subjects", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    volumes = out / "volumes"
    volumes.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    counts = {"CN": args.cn_subjects, "MCI": args.mci_subjects, "AD": args.ad_subjects}
    subject_counter = 0

    for diagnosis, n_subjects in counts.items():
        for _ in range(n_subjects):
            subject_counter += 1
            subject_id = f"SUBJ_{subject_counter:04d}"
            baseline_age = float(rng.uniform(58, 88))
            visits = 2 if diagnosis == "CN" and subject_counter % 6 == 0 else 1

            for visit in range(visits):
                age = baseline_age + 0.5 * visit
                scan_id = f"{subject_id}_V{visit + 1}"
                filename = f"{scan_id}.nii.gz"
                arr = synthetic_brain(age, diagnosis, seed=args.seed + subject_counter * 10 + visit)
                nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), volumes / filename)
                rows.append(
                    {
                        "subject_id": subject_id,
                        "scan_id": scan_id,
                        "file_path": f"volumes/{filename}",
                        "age": round(age, 2),
                        "diagnosis": diagnosis,
                        "sex": rng.choice(["M", "F"]),
                        "site": rng.choice(["SITE_A", "SITE_B", "SITE_C"]),
                    }
                )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(out / "metadata.csv", index=False)
    print(f"Created {len(metadata)} scans from {metadata['subject_id'].nunique()} subjects")
    print(metadata.groupby('diagnosis').agg(scans=('scan_id','size'), subjects=('subject_id','nunique')))
    print(f"Metadata: {out / 'metadata.csv'}")


if __name__ == "__main__":
    main()
