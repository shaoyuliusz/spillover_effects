import numpy as np
import pandas as pd
import random
import pytest

from spillover_effects.datapipes.make_graph import (
    make_bipartite_graph,
    make_sq_lattice_graph,
)
from spillover_effects.utils.dataset import Dataset, BipartiteDataset


def generate_treatment_vec(size, seed=123):
    """
    generate a random treatment vector of given size
    and potential simulated treatment vectors
    """
    np.random.seed(seed)
    random.seed(seed)
    tr_vec = np.random.randint(low=0, high=2, size=size)
    potential_tr_vec = np.array(
        [random.sample(list(tr_vec), len(tr_vec)) for _ in range(20)]
    )
    return tr_vec, potential_tr_vec


@pytest.fixture(
    params=[(9, "weight"), (9, None)],
    ids=["weighted_9", "unweighted_9"],
)
def generate_dataset(request):
    """Generate a Dataset object"""
    N, weight = request.param
    lattice_graph = make_sq_lattice_graph(N=N, weight=weight, seed=123)
    if weight:
        lattice_dataset = Dataset(graph=lattice_graph, edge_weight_attr=weight)
    else:
        lattice_dataset = Dataset(graph=lattice_graph, edge_weight_attr=None)

    tr_vec, potential_tr_vec = generate_treatment_vec(size=N, seed=56)

    return lattice_dataset, tr_vec, potential_tr_vec


@pytest.fixture(
    params=[(20, 10, None), (20, 10, "weight")],
    ids=["unweighted_20_10", "weighted_20_10"],
)
def generate_bipartite_dataset(request):
    n_out, n_div, weight = request.param
    bigraph = make_bipartite_graph(
        n_outcome=n_out, n_diversion=n_div, weight=weight, seed=123
    )
    bipartite_dataset = BipartiteDataset(graph=bigraph, edge_weight_attr=weight)

    tr_vec, potential_tr_vec = generate_treatment_vec(size=n_div + n_out, seed=56)
    return bipartite_dataset, tr_vec, potential_tr_vec
