import numpy as np
import pandas as pd
import networkx as nx
from typing import Callable


def make_corr_out(
    degree, degree2, correlate=True, include_degree2=True, seed=None
) -> np.ndarray:
    """
    This makes treatment effect positively correlated with the degree that a node has
    aka treatment heterogeneity.

    Parameters
    ----------
    degree : numpy.ndarray
        Array containing the degree of each node.
    degree2 : numpy.ndarray
        Array containing the secondary degree of each node.
    correlate : bool, optional
        A flag indicating whether to correlate the treatment effect with the degrees.
    include_degree2 : bool, optional
        A flag indicating whether to include degree2 in the calculation.
        Defaults to True.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    treatment_effect : numpy.ndarray
        (N, ) shaped array containing the treatment effect for each node

    References
    ----------
    Peter M. Aronow, Dean Eckles, Cyrus Samii, Stephanie Zonszein (2020).
    Spillover Effects in Experimental Data. https://arxiv.org/pdf/2001.05444.pdf
    """
    np.random.seed(seed)

    if correlate:
        if include_degree2:
            treatment_effect = (degree + degree2 + degree * degree2) * np.abs(
                np.random.normal(size=len(degree))
            )
        else:
            treatment_effect = degree * np.abs(np.random.normal(size=len(degree)))
        treatment_effect += np.random.normal(loc=1, scale=0.25, size=len(degree))
    else:
        treatment_effect = np.abs(np.random.normal(size=len(degree)))

    return treatment_effect


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


def make_dilated_baseline_continuous(graph: nx.Graph, **kwargs) -> np.ndarray:
    """
    make dilated potential outcome baseline y_i(00) for all i.

    Parameters
    ----------
    graph :
        The number of observations to simulate.

    **kwargs
        Additional keyword arguments to set non-default values for the parameters

    Returns:
    ----------
        potential_outcome_base: (n_node, ) shaped numpy.ndarray of the baseline outcome for each unit.
    """

    seed = kwargs.get("seed", 123)
    correlate = kwargs.get("correlate", True)
    deg2 = kwargs.get("include_degree2", True)

    adj_matrix = nx.adjacency_matrix(graph).toarray()
    degree = np.sum(adj_matrix, axis=1)  # number of adjacent nodes for each focal node

    nei2 = np.dot(adj_matrix, adj_matrix)
    nei2[nei2 > 1] = 1
    np.fill_diagonal(nei2, 0)
    degree2 = np.sum(nei2, axis=1)

    potential_outcome_base = make_corr_out(
        degree, degree2, correlate=correlate, include_degree2=deg2, seed=seed
    )

    return potential_outcome_base


def generate_ground_truth_grid(
    potential_outcome_base,
    grid,
    func_0: Callable[[float], float],
    func_1: Callable[[float], float],
) -> pd.DataFrame:
    """
    generate ground truth of average dose response E(Y(t, e)) at exposure level e for e of given grids, averaged over all units

    Args:
        potential_outcome_base
        grid: a numpy array of K exposure levels between 0 and 1.
        func_0: the functional form that the dose response function shape takes for t=0
        func_1: the functional form that the dose response function shape takes for t=1

    Returns:
        pandas dataframe with ground truth for average y(1, e) and y(0, e) for each exposure level e.
    """

    yt_ground_truth_1 = [
        np.mean(func_1(t) * potential_outcome_base) for idx, t in enumerate(grid)
    ]
    yt_ground_truth_0 = [
        np.mean(func_0(t) * potential_outcome_base) for idx, t in enumerate(grid)
    ]
    return pd.DataFrame(
        data={
            "grid": grid,
            "yt_true_0": yt_ground_truth_0,
            "yt_true_1": yt_ground_truth_1,
        }
    )


def generate_obs_outcome(
    potential_outcome_base: np.ndarray,
    treatment_vector: np.ndarray,
    exposure_vector: np.ndarray,
    func_0: Callable[[float], float],
    func_1: Callable[[float], float],
) -> np.ndarray:
    """
    generate observed outcomes given treatment, exposure and potential outcomes
    Args:
        potential_outcome_base: (N, )
        treatment_vector: (N, )
        exposure_vector: (N, )
    Returns:
        obs_outcome: (N, ) observed outcomes given treatment assignment and exposure.

    """
    # compute observed outcomes
    exposure_vector = exposure_vector.reshape(-1)
    treatment_vector = treatment_vector.reshape(-1)
    mask = treatment_vector == 0  # treated

    obs_outcome = potential_outcome_base.copy()
    obs_outcome[mask] *= func_0(exposure_vector[mask])
    obs_outcome[~mask] *= func_1(exposure_vector[~mask])

    return obs_outcome


def generate_obs_outcome_bipartite(
    potential_outcome_base, exposure_vector
) -> np.ndarray:
    """
    generate observed outcomes given exposure and baseline outcomes

    Args:
        potential_outcome_base: (N, ) this should be outcome node degrees if using Doudnichenko et al. (2020) simulation
        exposure_vector: (N, )

    Returns:
        obs_outcome: (N, ) observed outcomes given treatment assignment and exposure.
    """
    return potential_outcome_base * exposure_vector


def generate_ground_truth_grid_bipartite(potential_outcome_base, grid) -> pd.DataFrame:
    """
    generate ground truth of average dose response E(Y(e)) at level e for e of given grids, averaged over all units

    Args:
        potential_outcome_base
        grid: a numpy array of K exposure levels between 0 and 1.

    Returns:
        pandas dataframe with ground truth for each grid exposure e.
    """

    res = np.mean(np.outer(potential_outcome_base, grid), axis=0)
    return pd.DataFrame({"grid": grid, "yt_true": res})
    # np.mean(np.outer(np.array(degrees), np.linspace(0, 1, 11)), axis=0)
    # np.outer(np.mean(degrees), np.linspace(0, 1, 11))
