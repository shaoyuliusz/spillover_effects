import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from spillover_effects.datapipes.experiment_design import (
    permute_with_replacement,
    permute_wo_replacement,
    gen_treatment_assignment,
    gen_treatment_assignment_bipartite,
    gen_treatment_assignment_simulation,
    gen_treatment_assignment_simulation_bipartite,
)


@pytest.mark.parametrize(
    "set_seed_1, set_seed_2, resample_1, resample_2",
    [(123, 123, 1, 1), (123, 456, 1, 1), (123, 123, 10, 10), (123, 456, 1, 1)],
)
def test_permute_with_replacement(set_seed_1, set_seed_2, resample_1, resample_2):
    """Test for method make_tr_vec_permutation_w_rep random seed"""
    tr_array_1 = permute_with_replacement(
        N=5, p=0.5, R=resample_1, num_workers=1, seed=set_seed_1
    )
    tr_array_2 = permute_with_replacement(
        N=5, p=0.5, R=resample_2, num_workers=1, seed=set_seed_2
    )

    if set_seed_1 == set_seed_2:
        assert_array_equal(tr_array_1, tr_array_2)
    else:
        assert_raises(AssertionError, assert_array_equal, tr_array_1, tr_array_2)


# @pytest.mark.parametrize(
#     "set_seed_1, set_seed_2, resample_1, resample_2",
#     [(123, 123, 1, 1), (123, 456, 1, 1), (123, 123, 10, 10), (123, 456, 1, 1)],
# )
# def test_permute_wo_replacement(set_seed_1, set_seed_2, resample_1, resample_2):
#     """Test for method make_tr_vec_permutation_w_rep random seed"""
#     tr_array_1 = permute_wo_replacement(N=10, p=0.5, R=resample_1, seed=set_seed_1)
#     tr_array_2 = permute_wo_replacement(N=10, p=0.5, R=resample_2, seed=set_seed_2)

#     if set_seed_1 == set_seed_2:
#         assert_array_equal(tr_array_1, tr_array_2)
#     else:
#         assert_raises(AssertionError, assert_array_equal, tr_array_1, tr_array_2)


# @pytest.mark.parametrize(
#     "n_resample",
#     [5, 10],
# )
# def test_permute_wo_replacement_unique(n_resample):
#     array = permute_wo_replacement(N=10, p=0.5, R=100, seed=1)
#     unique_rows = np.unique(array, axis=0)
#     assert unique_rows.shape[0] == array.shape[0]


@pytest.mark.parametrize(
    "n_units, p_treatment",
    [(150, 1), (100, 0.5), (5, 0.0)],
)
def test_gen_treatment_assignment_return_shapes(n_units, p_treatment):
    array = gen_treatment_assignment(n_units, p_treatment, set_seed=123)
    assert all(element in [0, 1] for element in array)
    assert array.shape == (n_units,)


def test_gen_treatment_assignment_seed():
    array_1 = gen_treatment_assignment(10, 0.5, set_seed=123)
    array_2 = gen_treatment_assignment(10, 0.5, set_seed=123)
    assert_array_equal(array_1, array_2)


@pytest.mark.parametrize(
    "n_units, p_treatment, n_resample",
    [(150, 1, 5), (100, 0.5, 5), (5, 0.0, 5)],
)
def test_gen_treatment_assignment_simulation_return_shapes(
    n_units, p_treatment, n_resample
):
    array = gen_treatment_assignment_simulation(
        n_units, p_treatment, n_resample, num_workers=1, set_seed=123
    )
    assert np.all(np.logical_or(array == 0, array == 1))
    assert array.shape == (n_resample, n_units)


def test_gen_treatment_assignment_simulation_seed():
    array_1 = gen_treatment_assignment_simulation(10, 0.5, 5, 1, set_seed=123)
    array_2 = gen_treatment_assignment_simulation(10, 0.5, 5, 1, set_seed=123)
    assert_array_equal(array_1, array_2)


@pytest.mark.parametrize(
    "n_outcome, n_diversion, p_treatment, n_resample",
    [(15, 10, 1, 5), (100, 10, 0.5, 5), (5, 15, 0.0, 5)],
)
def test_gen_treatment_assignment_simulation_bipartite_return_shapes(
    n_outcome, n_diversion, p_treatment, n_resample
):
    array = gen_treatment_assignment_simulation_bipartite(
        n_outcome, n_diversion, p_treatment, n_resample, num_workers=1, set_seed=123
    )
    assert np.all(np.logical_or(array == 0, array == 1))
    assert array.shape == (n_resample, n_outcome + n_diversion)


def test_gen_treatment_assignment_simulation_bipartite_seed():
    array_1 = gen_treatment_assignment_simulation_bipartite(
        20, 10, 0.5, 5, 1, set_seed=123
    )
    array_2 = gen_treatment_assignment_simulation_bipartite(
        20, 10, 0.5, 5, 1, set_seed=123
    )
    assert_array_equal(array_1, array_2)
