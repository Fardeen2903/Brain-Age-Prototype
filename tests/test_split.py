import pandas as pd

from brain_age.split import split_cn_subjects


def test_subjects_never_cross_splits():
    rows = []
    for subject in range(30):
        for visit in range(2):
            rows.append(
                {
                    "subject_id": f"S{subject}",
                    "scan_id": f"S{subject}_V{visit}",
                    "file_path": "dummy.nii.gz",
                    "age": 60 + subject / 2,
                    "diagnosis": "CN",
                }
            )
    df = pd.DataFrame(rows)
    train, val, test = split_cn_subjects(df, seed=7)
    assert set(train.subject_id).isdisjoint(set(val.subject_id))
    assert set(train.subject_id).isdisjoint(set(test.subject_id))
    assert set(val.subject_id).isdisjoint(set(test.subject_id))
