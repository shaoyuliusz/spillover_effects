import numpy as np
import random
import networkx as nx


def make_graph(n, model, weight="weight", seed=None, **kwargs):
    """
    Factory function to build random graphs.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    model : str
        The type of random graph to generate. Options are:
            - "sq_lattice": Square lattice graph.
            - "scale_free": Barabasi-Albert scale-free graph.
            - "small_world": Watts-Strogatz small-world graph.
    weight : str, optional
        The attribute name for edge weights. Defaults to "weight".
    seed : int, optional
        Random seed for reproducibility.
    **kwargs:
        for small-world graph model:
        k : int, default=4
            Each node is joined with its `k` nearest neighbors in a ring topology.
        p : float, default=0.2
            The probability of rewiring each edge in the Watts-Strogatz small-world graph model.

    Returns
    -------
    networkx.Graph
        The generated random graph.

    Raises
    ------
    ValueError
        If an invalid model is specified.

    """
    if model == "sq_lattice":
        return make_sq_lattice_graph(n, weight, seed)
    elif model == "scale_free":
        return make_barabasi_graph(n, weight, seed)
    elif model == "small_world":
        k = kwargs.get("k", 4)
        p = kwargs.get("p", 0.2)
        return make_watts_strogatz_graph(n, k, p, weight, seed)
    else:
        raise ValueError("Invalid model specified")


def make_sq_lattice_graph(N, weight="weight", seed=None) -> nx.Graph:
    """
    Generate a circular lattice graph
    """
    np.random.seed(seed)
    if np.sqrt(N) != np.round(np.sqrt(N)):
        raise ValueError(f"N must be a square number, not {N}.")

    dim = int(np.sqrt(N))
    G = nx.generators.lattice.grid_2d_graph(dim, dim, periodic=True)

    if weight:
        for u, v in G.edges():
            # Generate random weight for each edge
            wei = np.random.uniform(low=1, high=2)  # adjust range as needed
            G[u][v][weight] = wei

    return G


def make_barabasi_graph(N, weight="weight", seed=None):
    """
    Returns a random graph using Barabási–Albert preferential attachment
    """
    np.random.seed(seed)
    G = nx.barabasi_albert_graph(N, 5, seed)
    while min(G.degree()) == 0:
        G = nx.barabasi_albert_graph(N, 5, seed)

    if weight:
        for u, v in G.edges():
            # Generate random weight for each edge
            wei = np.random.uniform(low=1, high=2)  # adjust range as needed
            G[u][v][weight] = wei
    return G


def make_watts_strogatz_graph(
    n: int, k: int, p: float, weight="weight", seed=None
) -> nx.Graph:
    """
    make a Watts Strogatz small world graph, make sure no unconnected units

    Args:
        n: number of nodes
        k: Each node is joined with its k nearest neighbors in a ring topology.
        p: The probability of rewiring each edge
        seed: indicator for random number generation state
        weight: edge weight name

    Returns:
        G: nx.Graph

    """
    np.random.seed(seed)
    alpha, beta = 2, 0.5

    G = nx.watts_strogatz_graph(n, k, p, seed)
    while min(G.degree()) == 0:
        G = nx.watts_strogatz_graph(n, k, p)

    if weight:
        for u, v in G.edges():
            # Generate random weight for each edge
            wei = np.random.beta(alpha, beta, size=1).item()
            G[u][v][weight] = wei

    return G


def make_bipartite_graph(n_outcome=1000, n_diversion=100, weight="weight", seed=None):
    """
    Create a bipartite graph where there are n_outcome outcome units and n_diversion diversion units.
    Similar to Causal Inference with Bipartite Designs

    Parameters:
        n_outcome: number of outcome units
        n_diversion: number of diversion units, or units we randomize on.
        weight: edge weight
        seed: indicator for random number generation state
    Returns:
        G: nx.Graph
    """
    random.seed(seed)
    np.random.seed(seed)

    if n_diversion < 10:
        raise ValueError("number of diversion units must be larger than 10. ")

    # Generate random values for m_i for each node in U
    m_values = [random.randint(1, 10) for _ in range(n_outcome)]

    # Create a bipartite graph
    G = nx.Graph()

    # Add nodes from sets U and V
    G.add_nodes_from(range(n_outcome), bipartite=0)  # outcome nodes
    G.add_nodes_from(
        range(n_outcome, n_outcome + n_diversion), bipartite=1
    )  # diversion nodes

    # Add edges between nodes
    for i, m_i in enumerate(m_values):
        v_set = np.random.choice(
            range(n_outcome, n_outcome + n_diversion), size=m_i, replace=False
        )
        for v in v_set:
            if weight is not None:
                G.add_edge(i, v, weight=1 / m_i)
            else:
                G.add_edge(i, v)
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
