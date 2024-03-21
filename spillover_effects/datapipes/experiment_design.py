"""
Module for making experiment design
"""

import random
import math
import numpy as np

from tqdm import tqdm
from typing import Dict, Any, Union
from joblib import Parallel, delayed, cpu_count


def permute_with_replacement(N: int, p: float, R: int, num_workers, seed):
    """
    more efficient algorithm to generate simulated assignment vectors with replacement.
    Args:
        N: number of units.
        p: proportion of units assigned to treatment.
        R: number of repetitions (treatment permutations).
            If allow_repetions = FALSE and R is bigger than the number of possible treatment assignments,
            then R is truncated to the number of possible treatment assignements.
        num_workers: number of workers if using parallel backend to speed up permutation
        seed: random number for result replicability.
    Returns:
        (N, ) or (R, N) matrix, each column is a simulated treatment vector.
    """
    n_treated = round(N * p)
    vec = np.concatenate(
        (np.ones(n_treated, dtype=int), np.zeros(N - n_treated, dtype=int))
    ).tolist()

    def gen_random(random_seed):
        if seed:
            random.seed(random_seed + seed)
        return random.sample(vec, len(vec))

    if num_workers == -1:
        num_workers = cpu_count()
    if num_workers > 1 and R > 1:
        random_list = Parallel(n_jobs=num_workers)(
            delayed(gen_random)(i) for i in range(R)
        )
    else:
        random_list = [gen_random(i) for i in range(R)]

    res = np.array(random_list)
    if res.shape[0] == 1:
        res = res.reshape(-1)

    return res


def permute_wo_replacement(N, p, R, seed=None):
    """
    generate simulated assignment vectors with replacement,
    it regenerates if a generated treatment vector is the same with a previous one.
    """
    random.seed(seed)
    n_treated = round(N * p)

    max_R = math.comb(N, n_treated)
    if R > max_R:
        R = max_R
        print(
            "R is larger than the number of possible treatment assignments. Truncating to",
            max_R,
        )

    tr_vec_sampled = np.zeros((R, N), dtype=int)

    for i in tqdm(range(R)):
        vec = np.concatenate(
            (np.ones(n_treated, dtype=int), np.zeros(N - n_treated, dtype=int))
        )
        np.random.shuffle(vec)

        while np.any(np.sum(np.vstack((vec, tr_vec_sampled[:i])), axis=0) > 1):
            np.random.shuffle(vec)

        tr_vec_sampled[i] = vec

    return tr_vec_sampled


def gen_treatment_assignment(n_units, p_treatment, set_seed=None):
    """generate treatment vectors
    Parameters:
        n_units: number of units to randomize
        p_treatment: probability of assigning to treatment group
        set_seed: sets the random seed number
    Returns:
        treatment_assignment: (n_units, ) shape vector
    """
    treatment_assignment = permute_with_replacement(
        N=n_units,
        p=p_treatment,
        R=1,
        num_workers=1,
        seed=set_seed,
    )
    return treatment_assignment


def gen_treatment_assignment_bipartite(
    n_diversion, n_outcome, p_treatment, set_seed=None
):
    """generate treatment vectors for bipartite network data
    Parameters:
        n_diversion: number of diversion units to randomize
        n_outcome: number of outcome units
        p_treatment: probability of assigning to treatment group
        set_seed: sets the random seed number
    Returns:
        treatment_assignment: (n_outcome+n_diversion, ) shape vector
    """
    treatment_assignment = permute_with_replacement(
        N=n_diversion, p=p_treatment, R=1, num_workers=1, seed=set_seed
    )
    treatment_assignment = np.concatenate(
        (np.zeros((n_outcome,)), treatment_assignment)
    )
    return treatment_assignment


def gen_treatment_assignment_simulation_bipartite(
    n_diversion, n_outcome, p_treatment, n_resample, num_workers, set_seed=None
) -> np.ndarray:
    """generate simulated treatment vectors for bipartite network data
    Parameters:
        n_diversion: number of diversion units to randomize
        n_outcome: number of outcome units
        set_seed: sets the random seed number
    Returns:
        treatment_assignment_simulation: (n_resample, n_outcome) shape matrix
    """
    treatment_assignment_simulation = permute_with_replacement(
        N=n_diversion,
        p=p_treatment,
        R=n_resample,
        num_workers=num_workers,
        seed=set_seed,
    )
    treatment_assignment_simulation = np.column_stack(
        (np.zeros((n_resample, n_outcome)), treatment_assignment_simulation)
    )
    return treatment_assignment_simulation


def gen_treatment_assignment_simulation(
    n_units, p_treatment, n_resample, num_workers, set_seed=None
):
    """generate simulated treatment vectors
    Parameters:
        n_units: number of units to randomize
        set_seed: sets the random seed number
    Returns:
        treatment_assignment_simulation: (n_resample, n_units) shape matrix
    """
    treatment_assignment_simulation = permute_with_replacement(
        N=n_units,
        p=p_treatment,
        R=n_resample,
        num_workers=num_workers,
        seed=set_seed,
    )

    return treatment_assignment_simulation
