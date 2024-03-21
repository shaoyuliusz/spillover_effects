import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from spillover_effects.datasets import make_cibd2020, make_pdcs2020
from spillover_effects.utils.dataloader import DesignDataLoader, ImputeDataLoader


def test_make_cibd2020_return_types():
    res_dataloader, df_ground_truth, obs_outcome = make_cibd2020(
        loader_method="design",
        n_outcome=100,
        n_diversion=10,
        p_treatment=0.5,
        n_resample=20,
        n_bins=10,
        seed=123,
    )
    assert isinstance(res_dataloader, DesignDataLoader)
    assert isinstance(df_ground_truth, pd.DataFrame)
    assert isinstance(obs_outcome, np.ndarray)


@pytest.mark.ci
@pytest.mark.parametrize(
    "loader_method_1, loader_method_2",
    [("impute", "impute"), ("impute", "design"), ("design", "design")],
)
def test_make_cibd2020_seed(loader_method_1, loader_method_2):
    dataloader1, df_ground_truth1, obs_outcome1 = make_cibd2020(
        loader_method=loader_method_1,
        n_outcome=100,
        n_diversion=10,
        p_treatment=0.5,
        n_resample=20,
        n_bins=10,
        seed=123,
    )
    dataloader2, df_ground_truth2, obs_outcome2 = make_cibd2020(
        loader_method=loader_method_2,
        n_outcome=100,
        n_diversion=10,
        p_treatment=0.5,
        n_resample=20,
        n_bins=10,
        seed=123,
    )

    assert_array_equal(dataloader1.tr_sim, dataloader2.tr_sim)
    assert_array_equal(dataloader1.exp_vec, dataloader2.exp_vec)
    assert_array_equal(obs_outcome1, obs_outcome2)
    assert_frame_equal(df_ground_truth1, df_ground_truth2)
