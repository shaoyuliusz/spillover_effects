import pytest
import numpy as np
import math

from spillover_effects.utils.dataloader import (
    DataLoader,
    ImputeDataLoader,
    DesignDataLoader,
)
from spillover_effects.utils.dataset import Dataset, BipartiteDataset
from spillover_effects.datapipes.make_graph import make_graph, make_bipartite_graph

from numpy.testing import assert_array_equal, assert_raises
from unittest.mock import Mock


@pytest.fixture
def example_data_vector():
    tr_sim_vector = np.array([1, 1, 0, 0])
    adj_matrix = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    expected_result = np.array([1.0, 0.5, 0.5, 0.0])
    return tr_sim_vector, adj_matrix, expected_result


@pytest.fixture
def example_data_matrix():
    tr_sim_vector = np.array([[1, 0, 1, 0], [0, 1, 1, 0]])
    adj_matrix = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    expected_result = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.5, 0.5, 1.0]])
    return tr_sim_vector, adj_matrix, expected_result


@pytest.fixture
def example_data1():
    tr_vector = np.array([1, 1, 0, 0, 0])
    exposure_vec = np.array([1, 0, 0.5, 0.3, 0.7])
    use_dosage = True
    n_grids = 6
    expected_bin_membership = np.array(
        [
            [False, False, False, False, True],
            [True, False, False, False, False],
            [False, False, True, False, False],
            [False, True, False, False, False],
            [False, False, False, True, False],
        ]
    )
    expected_t_e_membership = np.array(
        [
            [False, False, False, False, False, False, False, False, False, True],
            [False, False, False, False, False, True, False, False, False, False],
            [False, False, True, False, False, False, False, False, False, False],
            [False, True, False, False, False, False, False, False, False, False],
            [False, False, False, True, False, False, False, False, False, False],
        ]
    )
    return (
        tr_vector,
        exposure_vec,
        use_dosage,
        n_grids,
        expected_bin_membership,
        expected_t_e_membership,
    )


@pytest.fixture
def example_data2():
    tr_vector = np.array([1, 1, 0, 0, 0])
    exposure_vec = np.array([1, 0, 0.5, 0.3, 0.7])
    use_dosage = False
    expected_bin_membership = np.array(
        [[False, True], [True, False], [False, True], [False, True], [False, True]]
    )

    expected_t_e_membership = np.array(
        [
            [False, False, False, True],
            [False, False, True, False],
            [False, True, False, False],
            [False, True, False, False],
            [False, True, False, False],
        ]
    )
    return (
        tr_vector,
        exposure_vec,
        use_dosage,
        expected_bin_membership,
        expected_t_e_membership,
    )


@pytest.fixture
def sq_lattice_data():
    g = make_graph(n=9, model="sq_lattice", weight=None, seed=123)
    potential_tr_vector = np.array(
        [
            [0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
        ]
    )
    tr_vector = potential_tr_vector[0, :]
    return g, potential_tr_vector, tr_vector


def test_make_prob_dist_values(example_data_vector, example_data_matrix):
    """Test make_prob_dist method return values."""

    # Arrange
    mock_dataset = Mock(Dataset)
    mock_dataset.num_nodes = 10
    design_dataloader = DesignDataLoader(mock_dataset)

    tr_sim_vector, adj_matrix, expected_result = example_data_matrix
    # act and assert
    assert_array_equal(
        design_dataloader.make_prob_dist(adj_matrix, tr_sim_vector), expected_result
    )

    tr_vector, adj_matrix, expected_result = example_data_vector
    # act and assert
    assert_array_equal(
        design_dataloader.make_prob_dist(adj_matrix, tr_vector), expected_result
    )


def test_make_exposure_map():
    pass


def test_make_exposure_membership(example_data1, example_data2):

    # Arrange
    mock_dataset = Mock(Dataset)
    mock_dataset.num_nodes = 10
    dl = DesignDataLoader(mock_dataset)

    _, exposure_vec1, use_dosage1, n_grids, expected_bin_membership1, _ = example_data1
    _, exposure_vec2, use_dosage2, expected_bin_membership2, _ = example_data2

    res1 = dl._make_exposure_membership(exposure_vec1, use_dosage1, n_grids)
    assert_array_equal(res1, expected_bin_membership1)

    res2 = dl._make_exposure_membership(exposure_vec2, use_dosage2)
    assert_array_equal(res2, expected_bin_membership2)


def test_make_exposure_treatment_membership(example_data1, example_data2):
    # Arrange
    mock_dataset = Mock(Dataset)
    mock_dataset.num_nodes = 10
    dl = DesignDataLoader(mock_dataset)

    tr_vec1, _, _, _, expected_bin_membership1, expected_t_e_membership1 = example_data1
    tr_vec2, _, _, expected_bin_membership2, expected_t_e_membership2 = example_data2

    assert_array_equal(
        expected_t_e_membership1,
        dl._make_exposure_treatment_membership(tr_vec1, expected_bin_membership1),
    )
    assert_array_equal(
        expected_t_e_membership2,
        dl._make_exposure_treatment_membership(tr_vec2, expected_bin_membership2),
    )


def test_prepare_data_reg_no_dosage(generate_dataset):
    """test for prepare data"""
    dataset, tr_vec, potential_tr_vec = generate_dataset
    ds_dataloader = DesignDataLoader(dataset)
    ds_dataloader.prepare_data(
        t_assignment=tr_vec,
        t_assignment_sim=potential_tr_vec,
        use_dosage=False,
        n_grids=None,
    )

    assert len(ds_dataloader.exposure_dict) == 4
    assert list(ds_dataloader.exposure_probs.keys()) == [
        "I_exposure",
        "prob_exposure_k_k",
        "prob_exposure_k_l",
        "obs_prob_exposure_individual_kk",
    ]

    assert ds_dataloader.exposure_probs["I_exposure"].shape == (4, 9, 20)
    assert len(ds_dataloader.exposure_probs["prob_exposure_k_k"]) == 4
    assert len(ds_dataloader.exposure_probs["prob_exposure_k_l"]) == math.perm(4, 2)
    assert ds_dataloader.exposure_probs["obs_prob_exposure_individual_kk"].shape == (
        4,
        9,
    )
    assert_array_equal(ds_dataloader.tr_vec, tr_vec)
    assert_array_equal(ds_dataloader.tr_sim, potential_tr_vec)


def test_prepare_data_reg_dosage(generate_dataset):
    dataset, tr_vec, potential_tr_vec = generate_dataset
    ds_dataloader = DesignDataLoader(dataset)
    ds_dataloader.prepare_data(
        t_assignment=tr_vec,
        t_assignment_sim=potential_tr_vec,
        use_dosage=True,
        n_grids=10,
    )

    assert len(ds_dataloader.exposure_dict) == 18
    assert list(ds_dataloader.exposure_probs.keys()) == [
        "I_exposure",
        "prob_exposure_k_k",
        "prob_exposure_k_l",
        "obs_prob_exposure_individual_kk",
    ]

    assert ds_dataloader.exposure_probs["I_exposure"].shape == (18, 9, 20)
    assert len(ds_dataloader.exposure_probs["prob_exposure_k_k"]) == 18
    assert len(ds_dataloader.exposure_probs["prob_exposure_k_l"]) == math.perm(18, 2)
    assert ds_dataloader.exposure_probs["obs_prob_exposure_individual_kk"].shape == (
        18,
        9,
    )
    assert_array_equal(ds_dataloader.tr_vec, tr_vec)
    assert_array_equal(ds_dataloader.tr_sim, potential_tr_vec)


def test_prepare_data_bi_dosage(generate_bipartite_dataset):
    dataset, tr_vec, potential_tr_vec = generate_bipartite_dataset
    ds_dataloader = DesignDataLoader(dataset)
    ds_dataloader.prepare_data(
        t_assignment=tr_vec,
        t_assignment_sim=potential_tr_vec,
        use_dosage=True,
        n_grids=10,
    )
    K = 10 - 1
    assert len(ds_dataloader.exposure_dict) == K
    assert list(ds_dataloader.exposure_probs.keys()) == [
        "I_exposure",
        "prob_exposure_k_k",
        "prob_exposure_k_l",
        "obs_prob_exposure_individual_kk",
    ]

    # assert ds_dataloader.exposure_probs["I_exposure"].shape == (K, 9, 20)
    assert len(ds_dataloader.exposure_probs["prob_exposure_k_k"]) == K
    assert len(ds_dataloader.exposure_probs["prob_exposure_k_l"]) == math.perm(K, 2)
    assert ds_dataloader.exposure_probs["obs_prob_exposure_individual_kk"].shape == (
        K,
        20,
    )
    assert_array_equal(ds_dataloader.tr_vec, tr_vec)
    assert_array_equal(ds_dataloader.tr_sim, potential_tr_vec)


def test_prepare_data_bi_no_dosage(generate_bipartite_dataset):
    dataset, tr_vec, potential_tr_vec = generate_bipartite_dataset
    ds_dataloader = DesignDataLoader(dataset)
    with pytest.raises(NotImplementedError):
        ds_dataloader.prepare_data(
            t_assignment=tr_vec,
            t_assignment_sim=potential_tr_vec,
            use_dosage=False,
            n_grids=None,
        )
