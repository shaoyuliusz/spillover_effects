import numpy as np
from functools import partial
from spillover_effects.datapipes.make_dilated_out import generate_obs_outcome


def test_make_dilated_baseline_continuous():
    pass


def test_generate_ground_truth_grid():
    pass


def polynomial(x, a, b, c):
    return a + b * x + c * x**2


def test_generate_obs_outcome():

    poly_0 = partial(polynomial, a=1, b=1, c=1)
    poly_1 = partial(polynomial, a=2, b=2, c=2)
    exposure_vector = np.array([0.5, 0.2, 0.3, 0.8])
    treatment_vector = np.array([1, 1, 0, 0])
    potential_outcome_base = np.array([1, 2, 3, 4], dtype=float)

    obs_outcome = generate_obs_outcome(
        potential_outcome_base, treatment_vector, exposure_vector, poly_0, poly_1
    )

    expected_outcome = np.array(
        [
            1 * (2 + 2 * 0.5 + 2 * 0.5**2),
            2 * (2 + 2 * 0.2 + 2 * 0.2**2),
            3 * (1 + 0.3 + 0.3**2),
            4 * (1 + 0.8 + 0.8**2),
        ]
    )
    assert obs_outcome == expected_outcome
