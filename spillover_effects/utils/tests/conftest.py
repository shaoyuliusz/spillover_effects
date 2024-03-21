import numpy as np
import pandas as pd

import pytest

from spillover_effects.datapipes.make_graph import (
    make_bipartite_graph,
    make_sq_lattice_graph,
)
from spillover_effects.utils.dataset import Dataset, BipartiteDataset


@pytest.fixture(
    params=[(9, False), (16, False), (9, True), (16, True)],
    ids=["unweighted_9", "unweighted_16", "weighted_9", "weighted_16"],
)
def generate_dataset(request):
    """Generate a Dataset object"""
    N, weighted = request.param
    lattice_graph = make_sq_lattice_graph(N=N, weighted=weighted)
    if weighted:
        lattice_dataset = Dataset(graph=lattice_graph, edge_weight_attr="weight")
    else:
        lattice_dataset = Dataset(graph=lattice_graph, edge_weight_attr=None)
    return lattice_dataset


@pytest.fixture(
    params=[(100, 10, False), (5, 5, False), (100, 10, True), (5, 5, True)],
    ids=["unweighted_100_10", "unweighted_5_5", "weighted_100_10", "weighted_5_5"],
)
def generate_bipartite_dataset(request):
    n_out, n_div, weighted = request.param
    bigraph = make_bipartite_graph(n_outcome=n_out, n_diversion=n_div)
    bipartite_data = BipartiteDataset(graph=bigraph, edge_weight_attr="weight")
    return bipartite_data
