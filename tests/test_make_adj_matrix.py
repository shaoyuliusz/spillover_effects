from spillover_effects.datapipes.make_graph import make_adj_matrix_sq_lattice
import numpy as np


def test_make_adj_matrix_sq_lattice():
    """
    ...
    """
    adjacency_matrix = np.array(
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
    )

    assert make_adj_matrix_sq_lattice(9) == adjacency_matrix


def test_make_watts_strogatz_graph():
    pass


def test_make_confounded_watts_strogatz_graph():
    pass
    # test the distribution of nodes degrees
