"""
Script to run simulations

Example usage:

python3 spillover_effects/simulation.py --outcome_effects heterogeneous \
                      --n_simulations 100 \
                      --output_dir ./output \
                      --n_outcome 1000 \
                      --n_diversion 100 \
                      --p_treatment 0.5 \
                      --n_resample 10000 \
                      --n_grids 21 \
                      --lower_grid_constraint 0.0 \
                      --upper_grid_constraint 1.0

"""

from datapipes.make_graph import make_graph, make_bipartite_graph
from utils.dataset import BipartiteDataset

import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from joblib import Parallel, delayed
from functools import partial

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

from models.ht_estimator import DesignEstimator


def simulate_bipartite_design(
    outcome_effects,
    n_simulations,
    output_dir,
    seed=123,
    n_outcome=1000,
    n_diversion=100,
    p_treatment=0.5,
    n_resample=10000,
    n_grids=21,
    lower_grid_constraint=0,
    upper_grid_constraint=1,
):
    yT_out_list = []
    tau_out_list = []

    # fix a graph dataset
    bigraph = make_bipartite_graph(n_outcome, n_diversion, weight="weight", seed=seed)
    bi_dataset = BipartiteDataset(graph=bigraph, edge_weight_attr="weight")
    dataloader = DesignDataLoader(bi_dataset)

    degrees = bi_dataset.degree_outcome

    grid_exposure = np.linspace(lower_grid_constraint, upper_grid_constraint, n_grids)
    if outcome_effects == "heterogeneous":
        df_ground_truth = generate_ground_truth_grid_bipartite(
            potential_outcome_base=degrees, grid=grid_exposure
        )

    elif outcome_effects == "homogeneous":
        df_ground_truth = generate_ground_truth_grid_bipartite(
            potential_outcome_base=np.mean(degrees), grid=grid_exposure
        )

    # tau(1, 0)

    tau_true = (
        df_ground_truth.loc[df_ground_truth["grid"] == 1, "yt_true"].values[0]
        - df_ground_truth.loc[df_ground_truth["grid"] == 0, "yt_true"].values[0]
    )

    t_assignment_sim = gen_treatment_assignment_simulation_bipartite(
        n_diversion, n_outcome, p_treatment, n_resample, num_workers=-1, set_seed=seed
    )

    for i in tqdm(range(n_simulations)):
        # generate a new treatment vector
        t_assignment = gen_treatment_assignment_bipartite(
            n_diversion, n_outcome, p_treatment, set_seed=i
        )

        dataloader.prepare_data(t_assignment, t_assignment_sim, n_grids=n_grids)
        # exposure vector differs in each iteration
        exposure_vector = dataloader.exp_vec
        grid_exposure = dataloader.treatment_grid

        if outcome_effects == "heterogeneous":
            obs_outcome = generate_obs_outcome_bipartite(degrees, exposure_vector)
        elif outcome_effects == "homogeneous":
            constant_C = np.mean(degrees)
            obs_outcome = generate_obs_outcome_bipartite(constant_C, exposure_vector)

        ht_est = DesignEstimator(dataloader=dataloader, obs_outcome=obs_outcome)
        ht_est.fit().var_yT_ht_adjusted().tau_ht().var_tau_ht()

        tau_ht_ci_lower = ht_est.tau_ht - 1.96 * ht_est.var_tau_ht
        tau_ht_ci_upper = ht_est.tau_ht + 1.96 * ht_est.var_tau_ht

        if tau_true <= tau_ht_ci_upper and tau_true >= tau_ht_ci_lower:
            covered = True

        yT_out = {"yT_ht": ht_est.yT_ht, "var_yT_ht": ht_est.var_yT_ht}

        tau_out = {
            "tau_ht": ht_est.tau_ht,
            "var_tau_ht": ht_est.var_tau_ht,
            "tau_ht_ci_lower": tau_ht_ci_lower,
            "tau_ht_ci_upper": tau_ht_ci_upper,
            "tau_true": tau_true,
            "cover": covered,
        }

        yT_out_list.append(yT_out)
        tau_out_list.append(tau_out)

    yT_out = pd.concat(
        [pd.DataFrame.from_dict(yT_out_list[i]) for i in range(n_simulations)]
    )
    return yT_out, df_ground_truth, pd.DataFrame(tau_out_list)


import argparse
import os
import pandas as pd

# Import your function here


def main():
    parser = argparse.ArgumentParser(description="Run bipartite design simulation.")
    parser.add_argument(
        "--outcome_effects",
        type=str,
        default="heterogeneous",
        choices=["heterogeneous", "homogeneous"],
        help="Type of outcome effects: 'heterogeneous' or 'homogeneous'. Default is 'heterogeneous'.",
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=100,
        help="Number of simulations to run. Default is 100.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the output files. Default is './output'.",
    )
    parser.add_argument(
        "--n_outcome",
        type=int,
        default=1000,
        help="Number of outcome nodes. Default is 1000.",
    )
    parser.add_argument(
        "--n_diversion",
        type=int,
        default=100,
        help="Number of diversion nodes. Default is 100.",
    )
    parser.add_argument(
        "--p_treatment",
        type=float,
        default=0.5,
        help="Probability of treatment assignment. Default is 0.5.",
    )
    parser.add_argument(
        "--n_resample",
        type=int,
        default=10000,
        help="Number of resamples. Default is 10000.",
    )
    parser.add_argument(
        "--n_grids",
        type=int,
        default=21,
        help="Number of grids for exposure vector. Default is 21.",
    )
    parser.add_argument(
        "--lower_grid_constraint",
        type=float,
        default=0,
        help="Lower bound for exposure grid. Default is 0.",
    )
    parser.add_argument(
        "--upper_grid_constraint",
        type=float,
        default=1,
        help="Upper bound for exposure grid. Default is 1.",
    )
    # Add more arguments as needed

    args = parser.parse_args()

    # Run simulation
    yT_out, df_ground_truth, tau_out = simulate_bipartite_design(
        outcome_effects=args.outcome_effects,
        n_simulations=args.n_simulations,
        n_outcome=args.n_outcome,
        n_diversion=args.n_diversion,
        p_treatment=args.p_treatment,
        n_resample=args.n_resample,
        n_grids=args.n_grids,
        lower_grid_constraint=args.lower_grid_constraint,
        upper_grid_constraint=args.upper_grid_constraint,
        output_dir=args.output_dir,
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save output to directory
    yT_out.to_csv(os.path.join(args.output_dir, "yT_out.csv"), index=False)
    df_ground_truth.to_csv(
        os.path.join(args.output_dir, "df_ground_truth.csv"), index=False
    )
    tau_out.to_csv(os.path.join(args.output_dir, "tau_out.csv"), index=False)


if __name__ == "__main__":
    main()
