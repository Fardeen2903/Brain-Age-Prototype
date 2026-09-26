# Brain Age Prototype

A small, reproducible research scaffold for 3D MRI brain-age regression and Brain Age Gap (BAG) analysis.

The prototype intentionally separates **pipeline validation** from **scientific claims**. It can run end-to-end on synthetic NIfTI volumes, then later accept properly preprocessed real MRI without changing the training/evaluation code.

## What this repo demonstrates

- NIfTI loading and canonical orientation
- 3D volume intensity standardization and resizing
- subject-level CN train/validation/test splitting to prevent repeated-visit leakage
- CN-only normative age-model training
- a fast `tiny3d` smoke-test network
- a MONAI 3D ResNet-18 option
- MAE and Pearson-r evaluation
- raw Brain Age Gap: `predicted_age - chronological_age`
- age-bias correction fitted **only on CN validation data**
- CN/MCI/AD BAG summaries, ANOVA, and Cohen's d
- saved plots, predictions, split files, and model checkpoints

## Important limitation

The local prototype does **not** claim to reproduce the medical preprocessing in the paper. For real ADNI-like data, skull stripping, N4 bias-field correction, and MNI registration should be performed as an explicit preprocessing/QC stage before training. Resizing a NIfTI volume is not equivalent to anatomical registration.

## Quick start (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .

python scripts/make_synthetic_dataset.py
pytest -q
ruff check .
python -m brain_age.train --config configs/smoke.yaml
python -m brain_age.evaluate --config configs/smoke.yaml
```

Outputs appear in `outputs/smoke/`.

## Expected metadata format

```csv
subject_id,scan_id,file_path,age,diagnosis,sex,site
S001,S001_BL,volumes/S001_BL.nii.gz,71.2,CN,F,SITE_A
S002,S002_BL,volumes/S002_BL.nii.gz,75.4,MCI,M,SITE_B
S003,S003_BL,volumes/S003_BL.nii.gz,79.0,AD,F,SITE_A
```

Required columns are `subject_id`, `scan_id`, `file_path`, `age`, and `diagnosis`.

## Research discipline built into the prototype

1. The age model is trained only on CN scans.
2. Repeated scans from one subject cannot cross CN splits.
3. The age-bias correction is fitted on CN validation predictions, not the final test/clinical cohorts.
4. MCI/AD subjects are used only in downstream evaluation.
5. Raw and corrected BAG are both preserved for comparison.

## When the real code/data arrives

Do not immediately replace the advisor's pipeline. First map its components onto this scaffold:

- What defines a subject versus a scan?
- Is splitting subject-level?
- Which preprocessing steps are already completed?
- What exact input shape and intensity normalization are used?
- Is architecture selection based only on CN validation performance?
- Is BAG corrected for age bias?
- Which clinical/site covariates are available?
- Why do any samples fail preprocessing/QC?

Then reproduce the baseline before introducing extensions.
