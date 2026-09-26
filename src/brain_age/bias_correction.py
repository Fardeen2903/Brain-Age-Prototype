from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass
class AgeBiasCorrector:
    intercept: float
    slope: float

    @classmethod
    def fit(cls, chronological_age: np.ndarray, predicted_age: np.ndarray) -> "AgeBiasCorrector":
        x = np.asarray(chronological_age, dtype=float).reshape(-1, 1)
        y = np.asarray(predicted_age, dtype=float)
        model = LinearRegression().fit(x, y)
        slope = float(model.coef_[0])
        if abs(slope) < 1e-8:
            raise ValueError("Bias-correction slope is too close to zero.")
        return cls(intercept=float(model.intercept_), slope=slope)

    def correct_predicted_age(self, predicted_age: np.ndarray) -> np.ndarray:
        predicted_age = np.asarray(predicted_age, dtype=float)
        return (predicted_age - self.intercept) / self.slope

    def to_dict(self) -> dict:
        return asdict(self)
