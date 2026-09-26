import numpy as np

from brain_age.bias_correction import AgeBiasCorrector


def test_linear_bias_correction_recovers_age():
    age = np.array([60, 65, 70, 75, 80], dtype=float)
    predicted = 20 + 0.7 * age
    corrector = AgeBiasCorrector.fit(age, predicted)
    corrected = corrector.correct_predicted_age(predicted)
    assert np.allclose(corrected, age)
