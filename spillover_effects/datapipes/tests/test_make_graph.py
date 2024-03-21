import numpy as np
import networkx as nx
import pytest
from numpy.testing import assert_array_equal
from spillover_effects.datapipes.make_graph import (
    make_sq_lattice_graph,
    make_barabasi_graph,
    make_bipartite_graph,
    make_watts_strogatz_graph,
    make_graph,
)


def test_make_graph_invalid_model():
    with pytest.raises(ValueError):
        make_graph(10, None)


@pytest.mark.parametrize(
    "N, expected_adjacency_matrix",
    [
        (
            9,
            np.array(
                [
                    [0, 1, 1, 1, 0, 0, 1, 0, 0],
                    [1, 0, 1, 0, 1, 0, 0, 1, 0],
                    [1, 1, 0, 0, 0, 1, 0, 0, 1],
                    [1, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 1],
                    [0, 1, 0, 0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 0, 0, 1, 1, 1, 0],
                ]
            ),
        ),
    ],
)
def test_make_sq_lattice_graph_unweighted(N, expected_adjacency_matrix):
    """
    Test to make the lattice graph return results.
    """

    G = make_sq_lattice_graph(N=N, weight=None)
    assert len(G.nodes()) == N
    assert all(degree == 4 for node, degree in G.degree())
    assert isinstance(G, nx.Graph)
    assert_array_equal(nx.adjacency_matrix(G).toarray(), expected_adjacency_matrix)


def test_make_sq_lattice_graph_invalid_input_N():
    N = 6
    msg_square = f"N must be a square number, not {N}."
    with pytest.raises(ValueError, match=msg_square):
        _ = make_sq_lattice_graph(N)


def test_make_sq_lattice_graph_weighted():
    N = 16  # Example size of the lattice
    G = make_sq_lattice_graph(N, weight="weight")
    assert isinstance(G, nx.Graph)
    assert len(G.nodes()) == N
    assert all(
        degree == 4 for node, degree in G.degree()
    )  # Lattice graph has degree 4 for each node
    assert all("weight" in G[u][v] for u, v in G.edges())
    assert all(0 <= G[u][v]["weight"] for u, v in G.edges())


@pytest.mark.parametrize("N, seed", [(10, 123), (20, 456), (30, 789)])
def test_make_barabasi_graph_weighted(N, seed):
    G = make_barabasi_graph(N=N, weight="w", seed=seed)
    assert isinstance(G, nx.Graph)
    assert len(G.nodes()) == N
    assert all(degree > 0 for node, degree in G.degree())
    assert all("w" in G[u][v] for u, v in G.edges())
    assert all(0 <= G[u][v]["w"] for u, v in G.edges())


@pytest.mark.parametrize(
    "n, k, p, weight, seed",
    [
        (10, 2, 0.1, None, 1),
        (20, 3, 0.2, "wei", 3),
        (30, 4, 0.3, None, 5),
    ],
)
def test_make_watts_strogatz_graph(n, k, p, weight, seed):
    G = make_watts_strogatz_graph(n, k, p, weight, seed)
    assert isinstance(G, nx.Graph)
    assert len(G.nodes()) == n
    assert all(degree > 0 for node, degree in G.degree())

    if weight:
        assert all(weight in G[u][v] for u, v in G.edges())
        assert all(0 <= G[u][v][weight] for u, v in G.edges())


def test_make_bipartite_graph_seed():
    g1 = make_bipartite_graph(n_outcome=100, n_diversion=10, seed=123)
    g2 = make_bipartite_graph(n_outcome=100, n_diversion=10, seed=123)
    assert nx.utils.graphs_equal(g1, g2)


@pytest.mark.parametrize("n_outcome, n_diversion", [(10, 15), (20, 10)])
def test_make_bipartite_graph_returns(n_outcome, n_diversion):
    bi_graph = make_bipartite_graph(n_outcome, n_diversion)

    num_outcome_nodes = sum(
        1 for node, attrs in bi_graph.nodes(data=True) if attrs.get("bipartite") == 0
    )

    num_diversion_nodes = sum(
        1 for node, attrs in bi_graph.nodes(data=True) if attrs.get("bipartite") == 1
    )

    assert num_diversion_nodes == n_diversion
    assert num_outcome_nodes == n_outcome
    assert bi_graph.number_of_nodes() == n_outcome + n_diversion
