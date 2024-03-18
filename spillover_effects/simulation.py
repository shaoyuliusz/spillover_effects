import numpy as np
import pandas as pd
import networkx as nx

from joblib import Parallel, delayed
from functools import partial
from spillover_effects.datapipes.make_graph import make_graph
from spillover_effects.scratch.make_exposure_map_AS import make_exposure_map_AS
from make_tr_vec_perm import (
    make_tr_vec_permutation_wo_rep,
    make_tr_vec_permutation_w_rep,
)
from spillover_effects.scratch.make_exposure_prob import make_exposure_prob
from spillover_effects.scratch.estimates import estimates
from spillover_effects.datapipes.make_dilated_out import make_dilated_out_1, make_dilated_out_2, make_corr_out


def gen_one_simulation(adj_matrix, potential_outcome, N, p, seed):
    """generate one simulation"""

    # generate one treatment vector
    tr_vector = make_tr_vec_permutation_wo_rep(N=N, p=p, R=1, seed=seed)

    # Create treatment exposure conditions
    obs_exposure = make_exposure_map_AS(adj_matrix, tr_vector, hop=1)

    obs_outcome = np.sum(
        obs_exposure * potential_outcome, axis=1
    )  # (200,) vector of observed outcomes

    potential_tr_vector = make_tr_vec_permutation_w_rep(N=N, p=p, R=10000, seed=seed)

    obs_prob_exposure = make_exposure_prob(
        potential_tr_vector, adj_matrix, make_exposure_map_AS, {"hop": 1}
    )

    out_data = estimates(obs_exposure, obs_outcome, obs_prob_exposure)

    ytht = pd.DataFrame(
        data=out_data["yT_ht"].reshape(1, -1),
        columns=["dir_ind1", "isol_dir", "ind1", "no"],
    )
    yth = pd.DataFrame(
        data=out_data["yT_h"].reshape(1, -1),
        columns=["dir_ind1", "isol_dir", "ind1", "no"],
    )

    return ytht, yth


def output_simulation(n, k, p, model, confounding=False, n_sims=100, seed=None):
    # "confounded_small_world"
    rand_graph = make_graph(n, k, p, model)
    if confounding == False:
        potential_outcome = make_dilated_out_1(rand_graph, make_corr_out)
    else:
        potential_outcome = make_dilated_out_2(rand_graph, make_corr_out)

    adj_matrix = nx.adjacency_matrix(rand_graph).toarray()
    partial_gen_one_sim = partial(
        gen_one_simulation, adj_matrix, potential_outcome, n, p, seed
    )

    mc_sim = Parallel(n_jobs=-1, verbose=2)(
        delayed(partial_gen_one_sim)() for _ in range(int(n_sims))
    )

    return potential_outcome, mc_sim
