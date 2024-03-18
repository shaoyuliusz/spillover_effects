import numpy as np
import random
import networkx as nx


def make_graph(n, k, p, model, seed=None):
    """factory to build random graphs"""

    if model == "sq_lattice":
        return make_sq_lattice_graph(n)
    elif model == "scale_free":
        return make_barabasi_graph(n, seed)
    elif model == "small_world":
        return make_watts_strogatz_graph(n, k, p, seed)
    # elif model == "confounded_small_world":
    #    return make_confounded_watts_strogatz_graph(n, k, p, seed)
    ##elif model == "dcbm":
    ##   return make_adj_matrix_dcbm(N, seed)
    else:
        raise ValueError("Invalid model specified")


def make_sq_lattice_graph(N, weighted=False) -> nx.Graph:
    """
    Generate a circular lattice graph
    """
    if np.sqrt(N) != np.round(np.sqrt(N)):
        raise ValueError(f"N must be a square number, not {N}.")

    dim = int(np.sqrt(N))
    G = nx.generators.lattice.grid_2d_graph(dim, dim, periodic=True)

    if weighted:
        for u, v in G.edges():
            # Generate random weight for each edge
            # weight = np.random.uniform(low=1, high=2)  # adjust range as needed
            weight = 1
            G[u][v]["weight"] = weight

    return G


def make_barabasi_graph(N, seed):
    """
    Returns a random graph using Barabási–Albert preferential attachment
    """
    random.seed(seed)

    G = nx.barabasi_albert_graph(N, 5, seed)
    while min(G.degree()) == 0:
        G = nx.barabasi_albert_graph(N, 5, seed)
    return G


def make_watts_strogatz_graph(
    n: int, k: int, p: float, seed=None, weighted=False
) -> nx.Graph:
    """
    make a Watts Strogatz small world graph, make sure no unconnected units
    Args:
        n: number of nodes
        k: Each node is joined with its k nearest neighbors in a ring topology.
        p: The probability of rewiring each edge
        seed: indicator for random number generation state
    Returns:

    """
    alpha, beta = 2, 0.5

    np.random.seed(seed)
    G = nx.watts_strogatz_graph(n, k, p, seed)
    while min(G.degree()) == 0:
        G = nx.watts_strogatz_graph(n, k, p)

    if weighted:
        for u, v in G.edges():
            # Generate random weight for each edge
            weight = np.random.beta(
                alpha, beta, size=1
            ).item()  # adjust range as needed
            G[u][v]["weight"] = weight

    return G


# def make_confounded_watts_strogatz_graph(n, k, p, seed=None) -> nx.Graph:
#     """Returns a Watts–Strogatz small-world graph with a confounding covariate X

#     Parameters
#     ----------
#     n : int
#         The number of nodes
#     k : int
#         Each node is joined with its `k` nearest neighbors in a ring
#         topology.
#     p : float
#         The probability of rewiring each edge
#     seed : integer, random_state, or None (default)
#         Indicator of random number generation state.
#         See :ref:`Randomness<randomness>`.
#     """
#     np.random.seed(seed)

#     X = sorted(np.random.uniform(0, 1, size=n))

#     if k > n:
#         raise nx.NetworkXError("k>n, choose smaller k or larger n")

#     # If k == n, the graph is complete not Watts-Strogatz
#     if k == n:
#         return nx.complete_graph(n)

#     G = nx.Graph()
#     nodes = list(range(n))  # nodes are labeled 0 to n-1
#     # connect each node to k/2 neighbors
#     for j in range(1, k // 2 + 1):
#         targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
#         G.add_edges_from(zip(nodes, targets))

#     for i in list(G.nodes):
#         G.nodes[i]["covariate"] = X[i]

#     # rewire edges from each node
#     # loop over all nodes in order (label) and neighbors in order (distance)
#     # no self loops or multiple edges allowed
#     for j in range(1, k // 2 + 1):  # outer loop is neighbors
#         targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
#         # inner loop in node order
#         for u, v in zip(nodes, targets):
#             if np.random.rand() < p:
#                 w = np.random.choice(nodes, p=X / np.sum(X))
#                 # this makes probability of getting node higher for latter nodes

#                 # Enforce no self-loops or multiple edges
#                 while w == u or G.has_edge(u, w):
#                     w = np.random.choice(nodes, p=X / np.sum(X))
#                     if G.degree(u) >= n - 1:
#                         break  # skip this rewiring
#                 else:
#                     G.remove_edge(u, v)
#                     G.add_edge(u, w)

#     while min(G.degree()) == 0:
#         make_confounded_watts_strogatz_graph(n, k, p)
#     return G


def make_bipartite_graph(num_nodes_U=1000, num_nodes_V=100):
    """
    Create a bipartite graph where there are n_outcome outcome units and n_diversion diversion units.
    Similar to Causal Inference with Bipartite Designs
    """

    # Generate random values for m_i for each node in U
    m_values = [random.randint(1, 3) for _ in range(num_nodes_U)]

    # Create a bipartite graph
    G = nx.Graph()

    # Add nodes from sets U and V
    G.add_nodes_from(range(num_nodes_U), bipartite=0)  # U nodes
    G.add_nodes_from(
        range(num_nodes_U, num_nodes_U + num_nodes_V), bipartite=1
    )  # V nodes

    # Add edges between nodes in U and V
    for i, m_i in enumerate(m_values):
        v_set = np.random.choice(
            range(num_nodes_U, num_nodes_U + num_nodes_V), size=m_i, replace=False
        )
        for v in v_set:
            G.add_edge(i, v, weight=1/m_i)
    return G
