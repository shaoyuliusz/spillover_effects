import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from typing import Union


class HistogramLearner(BaseEstimator):
    def __init__(self, num_bins):
        self.num_bins = num_bins

    def fit(self, X: np.ndarray):
        """locate X in histograms
        Args:
            X: (R, 1) shaped array: exposure distribution of a given unit.

        Returns
            self : object
            Returns the instance itself.
        """
        hist, bin_edges = np.histogram(
            X, bins=self.num_bins, range=(0, 1), density=False
        )
        self.hist = hist / np.sum(hist)
        self.bin_edges = bin_edges
        return self

    def score_samples(self, exp_level: Union[float, np.ndarray]) -> np.ndarray:
        """compute the propensity score for the given exposure

        Parameters:
            exp_level: a scalar of exposure level or a (N, ) array specifying the exposure grids

        Returns:
            bin_score
                if scalar, then returns an array (1,) shape for its probability
                if (N, ) array, then returns (N, ) gps (probability of happening) at the specified grids
        """

        inds = np.digitize(exp_level, self.bin_edges, right=False)

        # deal with the right edge case
        if isinstance(inds, (int, np.integer)):
            if inds > len(self.hist):
                inds -= 1
        elif isinstance(inds, np.ndarray):
            right_edge = int(len(self.hist)) + 1
            inds[inds == right_edge] -= 1

        # compute probability in bins
        bin_score = self.hist[inds - 1].reshape(-1)
        return bin_score


class ReflectiveLearner(BaseEstimator):
    def __init__(self, bandwidth, kernel):
        self.learner = KernelDensity(kernel=kernel, bandwidth=bandwidth)

    def fit(self, X: np.ndarray):
        # if len(X.shape) == 2:
        #     X = X.reshape(-1)
        X_augmented = np.stack((-X, X, 2 - X))

        self.learner.fit(X_augmented.reshape(-1, 1))  # (300,1)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """compute the propensity score for the given X

        Args:
            X: array-like of shape (n_samples, n_features)
        Returns:
            grid_score (N,)

        """
        scores = self.learner.score_samples(X)
        return np.exp(scores) * 3
