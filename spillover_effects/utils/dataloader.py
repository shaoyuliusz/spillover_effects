import random
from tqdm import tqdm
import math
import numpy as np
from typing import Dict, Any, Union
from joblib import Parallel, delayed
from abc import ABC
from dataset import Dataset, BipartiteDataset
from gps_learners import HistogramLearner, ReflectiveLearner

_implemented_data_backends = ["Dataset", "BipartiteDataset"]

GPSLearner = Union[HistogramLearner, ReflectiveLearner]


class DataLoader(ABC):
    """Base Class DataLoader data-backends

    Parameters
    ----------
    data : :class:`Dataset` object
        The :class:`DoubleMLData` object providing the organized graph data.
    n_bins: number of treatment grids
    p_treatment: percentage of units assigned to treatment
    n_resample: number of resample times for estimating generalized propensity score
    num_workers: number of cores for parallel processing when generating treatment simulations


    """

    def __init__(
        self,
        data: Dataset,
        n_bins: int,
        p_treatment: float,
        n_resample: int,
        num_workers: Union[int, None],
    ):
        self.data = data
        # check data type
        if not isinstance(data, Dataset):
            raise TypeError(
                "The data must be of "
                + " or ".join(_implemented_data_backends)
                + " type. "
                f"{str(data)} of type {str(type(data))} was passed."
            )
        self._is_bipartite_data = False
        if isinstance(data, BipartiteDataset):
            self._is_bipartite_data = True
            self.n_outcome = self.data.n_outcome
            self.n_diversion = self.data.n_diversion

        self.n_units = self.data.num_nodes
        self.n_bins = n_bins
        self.treatment_grid = np.linspace(0, 1, n_bins + 1)
        self.p_treatment = p_treatment
        self.n_resample = n_resample
        if not num_workers:
            self.num_workers = 1
        else:
            self.num_workers = num_workers
        self._check_init_params()

    def _check_init_params(self):
        if not isinstance(self.n_bins, int):
            raise TypeError(
                f"n_bins parameter must be an integer, "
                f"but found type {type(self.n_bins)}"
            )
        if not isinstance(self.p_treatment, float):
            raise TypeError(
                f"p_treatment parameter must be a float number, "
                f"but found type {type(self.p_treatment)}"
            )
        if (self.p_treatment >= 1) or (self.p_treatment <= 0):
            raise ValueError(
                f"p_treatment parameter must be between 0 and 1,"
                f"but found value {self.p_treatment}"
            )
        if not isinstance(self.n_resample, int):
            raise TypeError(
                f"n_resample parameter must be an integer, "
                f"but found type {type(self.n_resample)}"
            )
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError(
                "num_workers option should be non-negative; "
                "use num_workers=0 to disable multiprocessing."
            )

    @staticmethod
    def make_tr_vec_permutation_w_rep(
        N: int, p: int | float, R: int, num_workers, seed
    ):
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
            (N, ) or (N, R) matrix
        """
        n_treated = round(N * p)
        vec = np.concatenate(
            (np.ones(n_treated, dtype=int), np.zeros(N - n_treated, dtype=int))
        ).tolist()

        def gen_random(random_seed):
            if seed:
                random.seed(random_seed + seed)
            return random.sample(vec, len(vec))

        if num_workers > 1:
            random_list = Parallel(n_jobs=num_workers)(
                delayed(gen_random)(i) for i in range(R)
            )
        else:
            random_list = [gen_random(i) for i in range(R)]

        res = np.array(random_list)
        if res.shape[0] == 1:
            res = res.reshape(-1)

        return res

    @staticmethod
    def make_tr_vec_permutation_wo_rep(N, p, R, seed=None):
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

    def gen_tr_vec(self, set_seed=None):
        """generate treatment vectors
        Parameters:
            set_seed: sets the random seed number
        Returns:
            tr_vector: (N, ) shape vector
        """
        if self._is_bipartite_data:
            tr_vector = self.make_tr_vec_permutation_w_rep(
                N=self.n_diversion,
                p=self.p_treatment,
                R=1,
                num_workers=1,
                seed=set_seed,
            )
            tr_vector = np.concatenate((np.zeros((self.n_outcome,)), tr_vector))
        else:
            tr_vector = self.make_tr_vec_permutation_w_rep(
                N=self.n_units,
                p=self.p_treatment,
                R=1,
                num_workers=1,
                seed=set_seed,
            )
        # self.tr_vector = tr_vector
        return tr_vector

    def gen_tr_sim(self, set_seed=None):
        """generate simulated treatment vectors
        Parameters:
            set_seed: sets the random seed number
        Returns:
            tr_sim_vectors: (N, R) shape matrix
        """
        if self._is_bipartite_data:
            tr_sim_vectors = self.make_tr_vec_permutation_w_rep(
                N=self.n_diversion,
                p=self.p_treatment,
                R=self.n_resample,
                num_workers=self.num_workers,
                seed=set_seed,
            )
            tr_sim_vectors = np.column_stack(
                (np.zeros((self.n_resample, self.n_outcome)), tr_sim_vectors)
            )
        else:
            tr_sim_vectors = self.make_tr_vec_permutation_w_rep(
                N=self.n_units,
                p=self.p_treatment,
                R=self.n_resample,
                num_workers=self.num_workers,
                seed=set_seed,
            )

        # self.tr_sim_vectors = self.tr_sim_vectors
        return tr_sim_vectors


class ImputeDataLoader(DataLoader):
    """
    exp_vec: (n_outcome, )
    exposure_gps: (n_outcome, )
    grid_treatment_gps: (11, n_outcome)
    """

    def __init__(
        self,
        data: Dataset,
        n_bins: int,
        p_treatment: float,
        n_resample: int,
        num_workers: Union[int, None],
        density_estimator: GPSLearner,
    ):
        super().__init__(data, n_bins, p_treatment, n_resample, num_workers)
        if not isinstance(density_estimator, ReflectiveLearner):
            raise NotImplementedError
        if not isinstance(density_estimator, GPSLearner):
            raise TypeError("density_estimator must be of GPSLearner type.")
        self.density_estimator = density_estimator(n_bins)
        self.N = self.n_outcome if self._is_bipartite_data else self.n_units
        self.tr_vec = None
        self.tr_sim = None
        self.exp_vec = None
        self.exposure_gps = None
        self.grid_treatment_gps = None

    def make_prob_dist(self, adj_matrix: np.ndarray, tr_vec: np.ndarray):
        """
        make probability distribution of exposure levels through large number of MC simulations

        Parameters:
            adj_matrix: N x N adjacency matrix of the graph
            tr_vector: a (N, ) binary vector or (1, N) matrix indicating treatment assignment, or a N x R matrix with R simulations

        Returns:
            if tr_vec is of shape (N, ), returns (N, ) shaped vector
            if tr_vec is of shape (N x R), returns a (R, N) matrix
            where each row is the exposure of N units for a particular treatment assignment.
        """

        exp_mat = np.dot(tr_vec, adj_matrix) / np.dot(np.ones_like(tr_vec), adj_matrix)
        if exp_mat.shape[0] == 1:
            exp_mat = exp_mat.reshape(-1)

        return exp_mat

    def estimate_gps(
        self,
        exposure_vector: np.ndarray,
        grid_treatment: np.ndarray,
        learner: list,
    ):
        """
        Estimate generalized propensity score for
        1) R_hat(i) the GPS for the given treatment
        2) R_hat(i, t) with t in a grid of treatment levels

        Parameters:
            exposure_vector: (N, ) shape exposure vector
            grid_treatment: a grid of treatment levels e.g. [0,0.1,0.2,...,1]
            learner:

        Returns:
            exposure_one_gps: (N, ) GPS estimates R_hat(i)
            grid_treatment_gps: (len(grid), N) GPS R_hat(i, t) with t in a grid of treatment levels
        """
        # initialize arrays

        exposure_one_gps = np.zeros((self.N,))
        grid_treatment_gps = np.zeros((len(grid_treatment), self.N))

        # for each column in dist_sim, apply learn_gps for that unit.
        for i in tqdm(range(0, self.N)):
            # estimated generalized propensity score for unit i given her realized level of treatment
            exposure_one_gps[i] = learner[i].score_samples(exposure_vector[i]).item()
            grid_treatment_gps[:, i] = learner[i].score_samples(grid_treatment)

        return exposure_one_gps, grid_treatment_gps

    def prepare_data(self, random_seed_tr=None, random_seed_sim=None):
        """
        prepare data for imputation-regression estimator
        """
        # compute treatment vector
        tr_vec = self.gen_tr_vec(set_seed=random_seed_tr)

        # compute treatment simulations
        if self.tr_sim is None:
            tr_sim = self.gen_tr_sim(set_seed=random_seed_sim)
        else:
            tr_sim = self.tr_sim

        adj_matrix = self.data.adj_matrix.toarray()
        # compute exposure vector
        dist_tr = self.make_prob_dist(adj_matrix, tr_vec)
        if self._is_bipartite_data:
            dist_tr = dist_tr[: self.n_outcome]

        # compute simulated exposure vectors
        dist_sim = self.make_prob_dist(adj_matrix, tr_sim)
        if self._is_bipartite_data:
            dist_sim = dist_sim[:, : self.n_outcome]

        # fit exposure density estimator on each outcome unit using simulated exposure vectors
        density_estimator_list = [
            self.density_estimator.fit(X=dist_sim[:, i]) for i in range(0, self.N)
        ]

        # estimate GPS for R_hat(i) and R_hat(i,t)
        exposure_gps, grid_treatment_gps = self.estimate_gps(
            exposure_vector=dist_tr,
            grid_treatment=self.treatment_grid,
            learner=density_estimator_list,
        )

        self.exposure_gps = exposure_gps
        self.grid_treatment_gps = grid_treatment_gps
        self.tr_vec = tr_vec
        self.tr_sim = tr_sim
        self.exp_vec = dist_tr


class DesignDataLoader(DataLoader):

    def __init__(self, data: Dataset, n_bins, p_treatment, n_resample, num_workers):
        super().__init__(data, n_bins, p_treatment, n_resample, num_workers)
        self.tr_vec = None
        self.tr_sim = None
        self.exposure_probs = None
        self.obs_exposure = None

    def make_obs_exposure(self, exposure_vec):
        """
        Args:
            exposure_vec: (N, ) exposure vector induced by a particular treatment vector
            n_bins: number of bins to discretize continuous exposure

        Returns:
            bin_membership: true/false matrix of shape (N, K)
        """
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_edges[-1] = 1.01
        # Create a boolean array indicating whether each element falls into each bin (1000, 10)
        bin_membership = (exposure_vec[:, np.newaxis] >= bin_edges[:-1]) & (
            exposure_vec[:, np.newaxis] < bin_edges[1:]
        )

        return bin_membership

    def make_exposure_prob(self, tr_vec, tr_sim_vec) -> tuple:
        """create a tuple of dictionaries of I_exposure, prob_exposure_k_k, prob_exposure_k_j:
        Args:
            tr_vec: (N, ) shape array of treatment assignment vector
            tr_sim_vec: (R, N) simulated treatment assignment vector

        Returns: a tuple:
        I_exposure: A list of K N*R numeric matrices of indicators,
          for whether units N are in exposure condition k
          over each of the possible R treatment assignment vectors.
          The number of numeric matrices K corresponds to the number of exposure conditions.

        prob_exposure_k_k: A list of K symmetric N*N numeric matrices,
          each containing individual exposure probabilities to condition k on the diagonal,
          and joint exposure probabilities to condition k on the off-diagonals.

        prob_exposure_k_j: A list of permutation(K,2) nonsymmetric N*N numeric matrices,
          each containing joint probabilities across exposure conditions k and l on the off-diagonal,
          and zeroes on the diagonal. When K = 4, the number of numeric matrices is 12;

        """
        adj_matrix = self.data.adj_matrix.toarray()
        dist_tr = self.make_prob_dist(adj_matrix, tr_vec)
        if self._is_bipartite_data:
            dist_tr = dist_tr[: self.n_outcome]

        # print(" self.n_outcome", self.n_outcome)
        N = self.n_outcome if self._is_bipartite_data else self.n_units
        K = self.n_bins
        R = self.n_resample
        exposure_names = ["t" + str(i) for i in range(K)]
        # a length K list of empty N x R matrices
        I_exposure = np.full((K, N, R), np.nan)

        # (R,N) matrix
        for i in tqdm(range(R)):
            # compute exposure vector
            exp_vec = self.make_prob_dist(adj_matrix, tr_sim_vec[i, :])
            if self._is_bipartite_data:
                exp_vec = exp_vec[: self.n_outcome]
            potential_exposure = self.make_obs_exposure(exp_vec)
            for j in range(K):  # 0...K. other entries are zeros
                I_exposure[j, :, i] = potential_exposure[:, j]

        # construct prob_exposure_k_k and prob_exposure_k_l
        prob_exposure_k_k = {}
        prob_exposure_k_j = {}

        # this can be parallelized
        # for i, exposure_k in enumerate(I_exposure.values()):
        #     exposure_k_name = exposure_names[i]
        #     iota_mat = np.eye(N)
        #     prob_exposure_k_k[exposure_k_name + "," + exposure_k_name] = (
        #         exposure_k @ exposure_k.T + iota_mat
        #     ) / (R + 1)

        #     for j, exposure_l in enumerate(I_exposure.values()):
        #         if i != j:
        #             exposure_l_name = exposure_names[j]
        #             prob_exposure_k_l[exposure_k_name + "," + exposure_l_name] = (
        #                 exposure_k.dot(exposure_l.T) / R
        #             )
        for i in tqdm(range(K)):
            exposure_k_name = exposure_names[i]
            iota_mat = np.eye(N)
            exposure_k = I_exposure[i, :, :]
            prob_exposure_k_k[exposure_k_name + "," + exposure_k_name] = (
                exposure_k @ exposure_k.T + iota_mat
            ) / (R + 1)

            for j in range(K):
                if i != j:
                    exposure_j_name = exposure_names[j]
                    exposure_j = I_exposure[j, :, :]
                    prob_exposure_k_j[exposure_k_name + "," + exposure_j_name] = (
                        exposure_k.dot(exposure_j.T) / R
                    )

        return I_exposure, prob_exposure_k_k, prob_exposure_k_j

    def make_prob_dist(self, adj_matrix, tr_vec):
        """
        make probability distribution of exposure levels through large number of MC simulations

        Parameters:
            adj_matrix: N x N adjacency matrix of the graph
            tr_vector: a (N, ) binary vector or (1, N) matrix indicating treatment assignment, or a N x R matrix with R simulations

        Returns:
            (N, ) or (R, N) matrix where each row is the exposure of N units for a particular treatment assignment.
        """

        exp_mat = np.dot(tr_vec, adj_matrix) / np.dot(np.ones_like(tr_vec), adj_matrix)
        if exp_mat.shape[0] == 1:
            exp_mat = exp_mat.reshape(-1)

        return exp_mat

    def make_prob_exposure_cond(self, prob_exposure_k_k: Dict[str, Dict[str, Any]]):
        """

        Args:
            prob_exposure_k_k: a dict of K symmetric N*N numeric matrices
        Returns:
            prob_exposure_cond: K x N matrix representing estimated π_i (d_k),
            each row is [π_1(d_k), π_2(d_k),...π_N(d_k)]

        """

        K = self.n_bins
        N = self.n_outcome if self._is_bipartite_data else self.n_units

        prob_exposure_cond = np.zeros((K, N))
        for j, exposure_name in enumerate(prob_exposure_k_k.keys()):
            prob_exposure_cond[j, :] = np.diag(prob_exposure_k_k[exposure_name])

        return prob_exposure_cond

    def prepare_data(self, random_seed_tr=None, random_seed_sim=None):
        """
        prepare data for design-based estimator
        """

        # compute treatment vector
        tr_vec = self.gen_tr_vec(set_seed=random_seed_tr)

        # compute treatment simulations
        if self.tr_sim is None:
            tr_sim = self.gen_tr_sim(set_seed=random_seed_sim)
        else:
            tr_sim = self.tr_sim

        # compute exposure probabilities dictionaries
        I_exposure, prob_exposure_k_k, prob_exposure_k_l = self.make_exposure_prob(
            tr_vec, tr_sim
        )

        # compute generalized propensity score π_i(d_k)
        obs_prob_exposure_individual_kk = self.make_prob_exposure_cond(
            prob_exposure_k_k
        )

        # compute exposure vector induced by treatment vector
        exp_vec = self.make_prob_dist(self.data.adj_matrix.toarray(), tr_vec)
        if self._is_bipartite_data:
            exp_vec = exp_vec[: self.n_outcome]

        # compute exposure in k conditions indicator matrix
        obs_exposure = self.make_obs_exposure(exp_vec)

        self.tr_vec = tr_vec
        self.tr_sim = tr_sim
        self.exp_vec = exp_vec
        self.exposure_probs = {
            "I_exposure": I_exposure,
            "prob_exposure_k_k": prob_exposure_k_k,
            "prob_exposure_k_l": prob_exposure_k_l,
            "obs_prob_exposure_individual_kk": obs_prob_exposure_individual_kk,
        }
        self.obs_exposure = obs_exposure

        return self
