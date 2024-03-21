import numpy as np
from spillover_effects.models.ht_estimator import DesignEstimator


def test_var_yT_ht_unadjusted():
    obs_exposure_test = np.array([[1, 0], [1, 0], [0, 1]])
    obs_outcome_test = np.array([-1, 2, 3])
    prob_exposure_test = {
        "prob_exposure_k_k": {
            "cond1,cond1": np.array(
                [[0.3, 0.2, 0.1], [0.2, 0.5, 0.4], [0.1, 0.4, 0.8]]
            ),
            "cond2,cond2": np.array(
                [[0.7, 0.4, 0.1], [0.4, 0.5, 0.1], [0.1, 0.1, 0.2]]
            ),
        }
    }
    expected = np.array([[9.11111], [180]])
    result = var_yT_ht_unadjusted(
        obs_exposure_test, obs_outcome_test, prob_exposure_test
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_var_yT_ht_A2_adjustment():
    obs_exposure_test = np.array([[1, 0], [1, 0], [0, 1]])
    obs_outcome_test = np.array([-1, 2, 3])
    prob_exposure_test = {
        "prob_exposure_k_k": {
            "cond1,cond1": np.array([[0.3, 0, 0], [0, 0.5, 0.4], [0, 0.4, 0.8]]),
            "cond2,cond2": np.array([[0.7, 0.4, 0.1], [0.4, 0.5, 0], [0.1, 0, 0.2]]),
        }
    }
    expected = np.array([[14.66667], [45]])
    result = var_yT_ht_A2_adjustment(
        obs_exposure_test, obs_outcome_test, prob_exposure_test
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_cov_yT_ht_adjusted():
    obs_exposure_test = np.array([[1, 0], [1, 0], [0, 1]])
    obs_outcome_test = np.array([-1, 2, 3])
    prob_exposure_test = {
        "prob_exposure_k_k": {
            "cond1,cond1": np.array(
                [[0.3, 0.2, 0.1], [0.2, 0.5, 0.4], [0.1, 0.4, 0.8]]
            ),
            "cond2,cond2": np.array(
                [[0.7, 0.4, 0.1], [0.4, 0.5, 0.1], [0.1, 0.1, 0.2]]
            ),
        },
        "prob_exposure_k_l": {
            "cond1,cond2": np.array([[0, 0.2, 0.4], [0.1, 0, 0.3], [0, 0.15, 0]]),
            "cond2,cond1": np.array([[0, 0.1, 0], [0.2, 0, 0.15], [0.4, 0.3, 0]]),
        },
    }
    expected = np.array([[-32.33333], [-32.33333]])
    result = cov_yT_ht_adjusted(obs_exposure_test, obs_outcome_test, prob_exposure_test)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
