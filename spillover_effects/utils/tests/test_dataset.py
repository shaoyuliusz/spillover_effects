import pytest
from spillover_effects.datapipes.make_graph import (
    make_bipartite_graph,
    make_sq_lattice_graph,
)
from spillover_effects.utils.dataset import Dataset, BipartiteDataset


@pytest.mark.ci
def test_dataset_summary():
    """Test dataset summary method"""
    not_graph = 100
    msg = (
        r"data must be of nx.Graph type. "
        f"{str(not_graph)} of type {str(type(not_graph))} was passed."
    )
    with pytest.raises(TypeError, match=msg):
        _ = Dataset(graph=100)

    # lattice unweighted graph
    lattice_graph = make_sq_lattice_graph(N=9, weighted=False)
    lattice_dataset = Dataset(graph=lattice_graph, edge_weight_attr=None)

    assert isinstance(lattice_dataset.__str__(), str)
    assert isinstance(lattice_dataset._data_summary_str(), str)
    assert isinstance(lattice_dataset._degree_summary(), str)
    assert isinstance(lattice_dataset._edge_summary(), str)


@pytest.mark.ci
def test_bipartite_dataset_count_units():
    """test count outcome and diversion units for bipartite dataset"""
    bigraph = make_bipartite_graph(n_outcome=100, n_diversion=30)
    bigraph_dataset = BipartiteDataset(bigraph)
    assert bigraph_dataset.n_outcome == 100
    assert bigraph_dataset.n_diversion == 30
    assert bigraph_dataset.num_nodes == 130
