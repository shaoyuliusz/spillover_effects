import pytest
import pandas as pd
import numpy as np
from functools import partial
from numpy.testing import assert_array_equal
from spillover_effects.datapipes.make_dilated_out import (
    make_corr_out,
    generate_obs_outcome,
    generate_obs_outcome_bipartite,
    generate_ground_truth_grid_bipartite,
    generate_ground_truth_grid,
)


@pytest.fixture
def random_data():
    degree = np.random.randint(1, 10, size=10)
    degree2 = np.random.randint(1, 10, size=10)
    return degree, degree2


def test_make_corr_out_include_degree2(random_data):
    degree, degree2 = random_data
    output = make_corr_out(degree, degree2, include_degree2=True)
    assert output.shape == degree.shape


def test_make_corr_out_exclude_degree2(random_data):
    degree, degree2 = random_data
    output = make_corr_out(degree, degree2, include_degree2=False)
    assert output.shape == degree.shape


def test_make_corr_out_seed(random_data):
    degree, degree2 = random_data
    output1 = make_corr_out(degree, degree2, seed=42)
    output2 = make_corr_out(degree, degree2, seed=42)
    assert np.array_equal(output1, output2)


def test_make_dilated_baseline_continuous():
    pass


@pytest.mark.parametrize("base", [10, np.array([1, 2, 3, 4, 5])])
def test_generate_ground_truth_grid_bipartite_return_types(base):
    grid = np.linspace(0, 1, 6)
    res = generate_ground_truth_grid_bipartite(base, grid)
    assert res.shape[0] == len(grid)
    assert res.shape[1] == 3
    assert isinstance(res, pd.DataFrame)


def test_generate_ground_truth_grid_bipartite_values():
    grid = np.linspace(0, 1, 6)
    res1 = generate_ground_truth_grid_bipartite(5, grid)
    assert_array_equal(
        np.array(res1["yt_true"].values), np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    )
    res2 = generate_ground_truth_grid_bipartite(np.array([1, 2, 3, 4, 5]), grid)
    assert_array_equal(
        np.array(res1["yt_true"].values), np.array([0.0, 0.6, 1.2, 1.8, 2.4, 3.0])
    )


def test_generate_obs_outcome_bipartite():
    """Test generate_obs_outcome_bipartite"""
    potential_outcome_base = np.array([1, 2, 3])
    exposure_vector = np.array([0.5, 0.7, 0.9])
    obs_outcome = generate_obs_outcome_bipartite(
        potential_outcome_base, exposure_vector
    )
    expected_obs_outcome = np.array([0.5, 1.4, 2.7])
    assert_array_equal(obs_outcome, expected_obs_outcome)


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
