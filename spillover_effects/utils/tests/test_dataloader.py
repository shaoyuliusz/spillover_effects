import pytest
import numpy as np
from spillover_effects.utils.dataloader import (
    DataLoader,
    ImputeDataLoader,
    DesignDataLoader,
)
from numpy.testing import assert_array_equal, assert_raises


class DummyDataLoader(DataLoader):
    def __init__(
        self,
        data,
        n_bins,
        p_treatment,
        n_resample,
        num_workers,
    ):
        DataLoader.__init__(self, data, n_bins, p_treatment, n_resample, num_workers)

    def prepare_data(self):
        return super().prepare_data()


@pytest.fixture
def gen_dummy_dataloader(generate_dataset):
    n_bins = 10
    p_treatment = 0.5
    n_resample = 10
    num_workers = 1
    dummy_dataloader = DummyDataLoader(
        generate_dataset, n_bins, p_treatment, n_resample, num_workers
    )
    return dummy_dataloader


def test_dataloader_base(gen_dummy_dataloader):
    """Test the abc dataloader class"""
    dummy_data = gen_dummy_dataloader
    assert dummy_data.n_bins == 10
    assert dummy_data.p_treatment == 0.5
    with pytest.raises(AttributeError):
        getattr(dummy_data, "n_outcome")
    with pytest.raises(AttributeError):
        getattr(dummy_data, "n_diversion")




# @pytest.fixture(params=[(1, 2), (3, 4)])
# def test_impute_loader_obj_num_nodes(generate_dataset, impute_dataloader_fixture):
#     assert generate_dataset.num_nodes == impute_dataloader_fixture.N


# def test_impute_loader_obj_tr_vec(impute_dataloader_fixture):
#     assert isinstance(impute_dataloader_fixture.tr_vec, np.ndarray)
#     assert impute_dataloader_fixture.tr_vec.shape == (impute_dataloader_fixture.N,)


# def test_make_prob_dist():
#     tr_vector = np.array([1, 1, 0, 0])
#     adj_matrix = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])

#     assert make_prob_dist(adj_matrix, tr_vector) == np.array([1.0, 0.5, 0.5, 0.0])
