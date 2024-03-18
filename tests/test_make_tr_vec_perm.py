import pytest
from spillover_effects.make_tr_vec_perm import (
    make_tr_vec_permutation_w_rep,
    make_tr_vec_permutation_wo_rep,
)


def test_make_tr_vec_permutation_w_rep():
    res_1 = make_tr_vec_permutation_w_rep(10, 0.3, 100, parallel=True, seed=1)
    res_2 = make_tr_vec_permutation_w_rep(10, 0.3, 100, parallel=True, seed=1)

    assert len(res_1) == len(res_2)
    assert all([a == b for a, b in zip(res_1, res_2)])


def test_make_tr_vec_permutation_wo_rep():
    res_1 = make_tr_vec_permutation_wo_rep(10, 0.3, 100, seed=1)
    res_2 = make_tr_vec_permutation_wo_rep(10, 0.3, 100, seed=1)

    assert len(res_1) == len(res_2)
    assert all([a == b for a, b in zip(res_1, res_2)])
