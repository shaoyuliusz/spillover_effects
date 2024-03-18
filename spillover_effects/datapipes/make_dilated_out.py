import numpy as np
import pandas as pd
import networkx as nx
from typing import Callable


def make_corr_out(degree, degree2, correlate, seed=None) -> np.ndarray:
    """
    this makes treatment effect positively correlated with the degree that a node has
    aka treatment heterogeneity
    """
    np.random.seed(seed)

    if correlate == "yes":
        return (degree + degree2 + degree * degree2) * np.abs(
            np.random.normal(size=len(degree))
        ) + np.random.normal(loc=1, scale=0.25, size=len(degree))
    elif correlate == "no":
        return np.abs(np.random.normal(size=len(degree)))


def make_dilated_out_1(
    graph, make_corr_out, multipliers=None, seed=None
) -> pd.DataFrame:
    """
    Returns:
        n_unit X K pandas dataframe
    """
    np.random.seed(seed)
    if multipliers is None:
        multipliers = [2, 1.5, 1.25]
    if len(multipliers) != 3:
        raise ValueError("Needs 3 multipliers")

    adj_matrix = nx.adjacency_matrix(graph).toarray()
    degree = np.sum(adj_matrix, axis=1)  # number of adjacent nodes for each focal node
    nei2 = np.dot(adj_matrix, adj_matrix)
    nei2[nei2 > 1] = 1
    np.fill_diagonal(nei2, 0)
    degree2 = np.sum(nei2, axis=1)

    baseline_out = make_corr_out(degree, degree2, "yes", seed=seed)
    potential_out = np.vstack(
        [
            multipliers[0] * baseline_out,
            multipliers[1] * baseline_out,
            multipliers[2] * baseline_out,
            baseline_out,
        ]
    ).T

    rownames = ["dir_ind1", "isol_dir", "ind1", "no"]
    return pd.DataFrame(data=potential_out, columns=rownames)


def make_dilated_out_2(
    graph: nx.Graph, make_corr_out, multipliers=None, seed=None
) -> pd.DataFrame:
    """
    Args:
        graph:
    Returns:
        n_unit X K pandas dataframe
    """
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    covar_values = list(nx.get_node_attributes(graph, "covariate").values())

    np.random.seed(seed)
    if multipliers is None:
        multipliers = [2, 1.5, 1.25]
    if len(multipliers) != 3:
        raise ValueError("Needs 3 multipliers")

    degree = np.sum(
        adj_matrix, axis=1
    )  # an array of number of adjacent nodes for each focal node
    nei2 = np.dot(adj_matrix, adj_matrix)
    nei2[nei2 > 1] = 1
    np.fill_diagonal(nei2, 0)
    degree2 = np.sum(nei2, axis=1)

    baseline_out = make_corr_out(degree, degree2, "yes", seed=seed)  # length=degree

    potential_out = np.vstack(
        [
            multipliers[0]
            * (baseline_out + np.random.normal(5, 1, size=len(degree)) * covar_values),
            multipliers[1]
            * (baseline_out + np.random.normal(5, 1, size=len(degree)) * covar_values),
            multipliers[2]
            * (baseline_out + np.random.normal(5, 1, size=len(degree)) * covar_values),
            baseline_out + np.random.normal(5, 1, size=len(degree)) * covar_values,
            covar_values,
        ]
    ).T

    rownames = ["dir_ind1", "isol_dir", "ind1", "no", "X"]
    return pd.DataFrame(data=potential_out, columns=rownames)


def make_dilated_baseline_continuous(
    graph: nx.Graph, make_corr_out: Callable, seed=None
) -> pd.DataFrame:
    """
    make dilated potential outcome baseline y_i(00) for all i.

    Returns:

    """
    np.random.seed(seed)

    adj_matrix = nx.adjacency_matrix(graph).toarray()
    degree = np.sum(adj_matrix, axis=1)  # number of adjacent nodes for each focal node
    nei2 = np.dot(adj_matrix, adj_matrix)
    nei2[nei2 > 1] = 1
    np.fill_diagonal(nei2, 0)
    degree2 = np.sum(nei2, axis=1)

    baseline_out = make_corr_out(degree, degree2, "yes", seed=seed)

    return baseline_out


def polynomial(x, a, b, c):
    return a * x**2 + b * x + c


def sigmoid(x, a, k):
    """k controls kink, larger k means larger kink"""
    return 1 / (1 + np.exp(-k * (x - a)))


def generate_ground_truth_grid(
    potential_outcome_base,
    grid,
    func_0: Callable[[float], float],
    func_1: Callable[[float], float],
) -> pd.DataFrame:
    """
    generate ground truth of E(Y(t)) at level t for t of given grids
    Args:
        potential_outcome_base
        grid
        func_0: the functional form that the dose response function shape takes for t=0
        func_1: the functional form that the dose response function shape takes for t=1
    Returns:
        pandas dataframe with ground truth for average y(t, 1) and y(t, 0) for each t.
    """

    yt_ground_truth_1 = [
        np.mean(func_1(t) * potential_outcome_base) for idx, t in enumerate(grid)
    ]
    yt_ground_truth_0 = [
        np.mean(func_0(t) * potential_outcome_base) for idx, t in enumerate(grid)
    ]
    return pd.DataFrame(
        data={"yt_true_0": yt_ground_truth_0, "yt_true_1": yt_ground_truth_1}
    )


def generate_obs_outcome(
    potential_outcome_base,
    treatment_vector,
    exposure_vector,
    func_0: Callable[[float], float],
    func_1: Callable[[float], float],
):
    """
    generate observed outcomes given treatment vector and potential outcomes
    Args:
        potential_outcome_base:
        treatment_vector:
        exposure_vector:
    Returns:
        obs_outcome: observed outcomes
    ******Note the input must be (K, ) vectors
    """
    # compute observed outcomes
    exposure_vector = exposure_vector.reshape(-1)
    treatment_vector = treatment_vector.reshape(-1)
    mask = treatment_vector == 0  # treated

    obs_outcome = potential_outcome_base.copy()
    obs_outcome[mask] *= func_0(exposure_vector[mask])
    obs_outcome[~mask] *= func_1(exposure_vector[~mask])

    return obs_outcome
