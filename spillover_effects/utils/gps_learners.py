import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity


class HistogramLearner(BaseEstimator):
    def __init__(self, num_bins):
        self.num_bins = num_bins

    def fit(self, X: np.ndarray):
        """locate X in histograms

        Args:
        X: (N, 1)

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

    def score_samples(self, X) -> np.ndarray:
        """compute the propensity score for the given X

        Args:
            X: (N,)
        Returns:
            grid_score (N,)
        """

        inds = np.digitize(X, self.bin_edges, right=True)
        # inds[-1] -= 1
        grid_score = self.hist[inds - 1].reshape(-1)
        return grid_score


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
