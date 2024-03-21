from spillover_effects.utils.dataloader import (
    DataLoader,
    DesignDataLoader,
    ImputeDataLoader,
)
from utils.gps_learners import HistogramLearner


def manual_impute_dataloader(
    dataset,
    n_bins,
    p_treatment,
    n_resample,
    num_workers,
    density_estimator,
    seed_tr,
    seed_sim,
):
    impute_loader_obj = ImputeDataLoader(
        dataset, n_bins, p_treatment, n_resample, num_workers, density_estimator
    )
    impute_loader_obj.prepare_data(seed_tr, seed_sim)
    return impute_loader_obj


def manual_design_dataloader(
    dataset,
    n_bins,
    p_treatment,
    n_resample,
    num_workers,
    seed_tr,
    seed_sim,
):
    design_loader_obj = DesignDataLoader(
        dataset, n_bins, p_treatment, n_resample, num_workers
    )
    design_loader_obj.prepare_data(seed_tr, seed_sim)
    return design_loader_obj
