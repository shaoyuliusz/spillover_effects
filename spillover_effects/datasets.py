"""Module for generating some sample datasets.  """

import pandas as pd
import numpy as np
import logging
from functools import partial
from datapipes.make_dilated_out import (
    make_dilated_baseline_continuous,
    generate_ground_truth_grid,
    generate_ground_truth_grid_bipartite,
    generate_obs_outcome,
    generate_obs_outcome_bipartite,
)

from datapipes.make_graph import make_graph, make_bipartite_graph
from datapipes.experiment_design import (
    gen_treatment_assignment,
    gen_treatment_assignment_bipartite,
    gen_treatment_assignment_simulation,
    gen_treatment_assignment_simulation_bipartite,
)

from utils.polynomials import quadratic, sigmoid, middle_points
from utils.dataset import Dataset, BipartiteDataset
from utils.dataloader import ImputeDataLoader, DesignDataLoader
from utils.gps_learners import HistogramLearner

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def make_pdcs2020(
    loader_method,
    n_units=100,
    p_treatment=0.5,
    n_resample=10000,
    n_grids=None,
    treatment_grid=middle_points(np.linspace(0, 1, 11)),
    n_bins=10,
    graph_model="small_world",
    **kwargs
):
    """
    Generate spillover network dataset
    """

    # get kwargs
    func_0 = kwargs.get("func_0", partial(quadratic, a=1, b=1, c=1))
    func_1 = kwargs.get("func_1", partial(quadratic, a=1, b=1, c=1))
    seed = kwargs.get("seed", 123)

    correlate = kwargs.get("correlate", True)
    include_degree2 = kwargs.get("include_degree2", True)

    # generate graph
    small_world_graph = make_graph(
        n=n_units, model=graph_model, weight="weight", seed=seed
    )
    small_world_dataset = Dataset(small_world_graph, edge_weight_attr="weight")

    # generate treatment and simulation assignments from experiment design
    t_assignment = gen_treatment_assignment(n_units, p_treatment, set_seed=seed)
    t_assignment_sim = gen_treatment_assignment_simulation(
        n_units, p_treatment, n_resample, num_workers=-1, set_seed=123
    )

    if loader_method == "impute":
        dataloader = ImputeDataLoader(small_world_dataset)
        dataloader.prepare_data(
            t_assignment,
            t_assignment_sim,
            n_grids=n_grids,
            treatment_grid=treatment_grid,
            n_bins=n_bins,
            density_estimator=HistogramLearner,
        )
    elif loader_method == "design":
        dataloader = DesignDataLoader(small_world_dataset)
        dataloader.prepare_data(t_assignment, t_assignment_sim, n_grids=n_grids)

    exposure_vector = dataloader.exp_vec
    grid_exposure = dataloader.treatment_grid

    # make baseline values for y
    potential_outcome_base = make_dilated_baseline_continuous(
        small_world_graph,
        seed=seed,
        correlate=correlate,
        include_degree2=include_degree2,
    )

    # compute ground truth at given levels of exposure grids
    df_ground_truth = generate_ground_truth_grid(
        potential_outcome_base, grid_exposure, func_0, func_1
    )

    # generate observed outcomes
    obs_outcome = generate_obs_outcome(
        potential_outcome_base, t_assignment, exposure_vector, func_0, func_1
    )

    return dataloader, df_ground_truth, obs_outcome


def make_cibd2020(
    loader_method,
    n_outcome=1000,
    n_diversion=100,
    p_treatment=0.5,
    n_resample=10000,
    n_grids=None,
    treatment_grid=middle_points(np.linspace(0, 1, 11)),
    n_bins=10,
    outcome_effects="heterogeneous",
    seed=123,
):

    bigraph = make_bipartite_graph(n_outcome, n_diversion, weight="weight", seed=seed)
    bi_dataset = BipartiteDataset(graph=bigraph, edge_weight_attr="weight")
    degrees = bi_dataset.degree_outcome

    # logger.info("dataset summary")
    # logger.info(print(bi_dataset))

    # generate treatment and simulation assignments from experiment design
    t_assignment = gen_treatment_assignment_bipartite(
        n_diversion, n_outcome, p_treatment, set_seed=seed
    )
    t_assignment_sim = gen_treatment_assignment_simulation_bipartite(
        n_diversion, n_outcome, p_treatment, n_resample, num_workers=-1, set_seed=seed
    )

    if loader_method == "impute":
        dataloader = ImputeDataLoader(bi_dataset)

        dataloader.prepare_data(
            t_assignment,
            t_assignment_sim,
            n_grids=n_grids,
            treatment_grid=treatment_grid,
            n_bins=n_bins,
            density_estimator=HistogramLearner,
        )
    elif loader_method == "design":
        dataloader = DesignDataLoader(bi_dataset)
        dataloader.prepare_data(t_assignment, t_assignment_sim, n_grids=n_grids)

    exposure_vector = dataloader.exp_vec
    grid_exposure = dataloader.treatment_grid

    if outcome_effects == "heterogeneous":
        df_ground_truth = generate_ground_truth_grid_bipartite(
            potential_outcome_base=degrees, grid=grid_exposure
        )
        obs_outcome = generate_obs_outcome_bipartite(degrees, exposure_vector)
    elif outcome_effects == "homogeneous":
        constant_C = np.mean(degrees)
        df_ground_truth = generate_ground_truth_grid_bipartite(
            potential_outcome_base=np.mean(degrees), grid=grid_exposure
        )
        obs_outcome = generate_obs_outcome_bipartite(constant_C, exposure_vector)

    return dataloader, df_ground_truth, obs_outcome
