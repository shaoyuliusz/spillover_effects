import pytest
import numpy as np
from spillover_effects.scratch.make_gps import (
    make_prob_dist,
    HistogramLearner,
    ReflectiveLearner,
)

NUM_BINS = 10


def test_make_prob_dist():
    tr_vector = np.array([1, 1, 0, 0])
    adj_matrix = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])

    assert make_prob_dist(adj_matrix, tr_vector) == np.array([1.0, 0.5, 0.5, 0.0])


def test_histogram_learner():
    X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    X_test = np.array([0.01, 0.1, 0.59, 0.61])
    hist_learner = HistogramLearner(num_bins=NUM_BINS)
    hist_learner.fit(X)

    assert hist_learner.bin_edges == np.linspace(0, 1, 11)
    assert hist_learner.hist == np.array(
        [0.0, 0.2, 0.4, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0]
    )
    grid_score = hist_learner.score_samples(X_test)
    assert grid_score == np.array([0.0, 0.2, 0.2, 0.0])
