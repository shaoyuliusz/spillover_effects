import pytest
import numpy as np
from spillover_effects.utils.gps_learners import HistogramLearner
from numpy.testing import assert_array_equal

NUM_BINS = 10


@pytest.fixture
def exposure_training_data():
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5])


@pytest.fixture(params=[np.array([0, 0.01, 0.1, 0.11, 0.19, 0.20, 0.21]), 0.15])
def exposure_to_score_data(request):
    return request.param


@pytest.fixture
def histogram_fitted_model(exposure_training_data):
    hist_learner = HistogramLearner(num_bins=NUM_BINS)
    hist_learner.fit(exposure_training_data)
    return hist_learner


def test_histogram_learner_attrs(histogram_fitted_model):
    """Tests attributes of fitted histogram learner"""
    assert_array_equal(histogram_fitted_model.bin_edges, np.linspace(0, 1, 11))
    assert_array_equal(
        histogram_fitted_model.hist,
        np.array([0.0, 0.2, 0.4, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0]),
    )


def test_histogram_learner_method(histogram_fitted_model, exposure_to_score_data):
    grid_score = histogram_fitted_model.score_samples(exposure_to_score_data)
    if isinstance(exposure_to_score_data, np.ndarray):
        expected_result = np.array([0.0, 0.0, 0.2, 0.2, 0.2, 0.4, 0.4])
        assert_array_equal(grid_score, expected_result)
    else:
        expected_result = 0.2
        assert grid_score == expected_result
