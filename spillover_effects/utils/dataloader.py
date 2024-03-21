import numpy as np

from tqdm import tqdm
from typing import Dict, Any, Union
from abc import ABC, abstractmethod

from .gps_learners import ReflectiveLearner
from .dataset import Dataset, BipartiteDataset

_implemented_data_backends = ["Dataset", "BipartiteDataset"]


class DataLoader(ABC):
    """Base Class DataLoader data-backends

    Parameters
    ----------
    data : :class:`Dataset` object
        The :class:`DoubleMLData` object providing the organized graph data.
    n_grids: number of treatment grids
    p_treatment: percentage of units assigned to treatment
    n_resample: number of resample times for estimating generalized propensity score
    num_workers: number of cores for parallel processing when generating treatment simulations
    """

    def __init__(
        self,
        data: Dataset,
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
        self._check_init_params()

    def _check_init_params(self):
        if not isinstance(self.data, Dataset):
            raise TypeError(
                f"data parameter must be of Dataset type, "
                f"but found type {type(self.data)}."
            )

    @abstractmethod
    def prepare_data(self):
        pass


class ImputeDataLoader(DataLoader):
    """
    DataLoader for imputation-based estimators

    Parameters
    ----------

    lower_grid_constraint:  float, optional(default = 0.05)
        This adds an optional constraint of the lower side of the treatment grid.
        This constraint is used in treatment_grids needed by learner.score_samples,
        n_bins=n_grids-1
        where the computation needs inds = np.digitize(np.linspace(lower,upper,n_bins), bin_edges, right=False),
        make it robust to right=False or right=True

    upper_grid_constraint: float, optional (default = 0.95)
        See above parameter. Just like above, but this is an upper constraint.

    density_estimator: estimator for GPS using empirical distribution of exposures.

    Attributes
    ----------

    exp_vec: (n_outcome, )
    exposure_gps: (n_outcome, )
    grid_treatment_gps: (11, n_outcome)

    """

    def __init__(
        self,
        data: Dataset,
        lower_grid_constraint=0.0,
        upper_grid_constraint=1.0,
    ):
        super().__init__(data)

        self.density_estimator = None
        self.n_grids = None
        self.n_bins = None
        self.N = self.n_outcome if self._is_bipartite_data else self.n_units
        self.tr_vec = None
        self.tr_sim = None
        self.exp_vec = None
        self.exposure_gps = None
        self.grid_treatment_gps = None
        self.lower_grid_constraint = lower_grid_constraint
        self.upper_grid_constraint = upper_grid_constraint

    def make_prob_dist(self, adj_matrix: np.ndarray, tr_vec: np.ndarray):
        """
        make probability distribution of exposure levels through large number of MC simulations

        Parameters:
            adj_matrix: N x N adjacency matrix of the graph
            tr_vector: a (N, ) binary vector or (1, N) matrix indicating treatment assignment, or a R x N matrix with R simulations

        Returns:
            if tr_vec is of shape (N, ), returns (N, ) shaped vector
            if tr_vec is of shape (R, N), returns a (R, N) matrix
            where each row is the exposure of N units for a particular treatment assignment.
        """

        exp_mat = np.dot(tr_vec, adj_matrix) / np.dot(np.ones_like(tr_vec), adj_matrix)
        if exp_mat.shape[0] == 1:
            exp_mat = exp_mat.reshape(-1)

        return exp_mat

    def make_summary(self):
        """make summary of the ImputeDataLoader object"""

        den_est = f"Density Estimator: {self.density_estimator}\n"
        if self._is_bipartite_data:
            num_units = (
                f"Total number of units: {self.n_outcome+self.n_diversion}\n",
                f"Number of outcome units: {self.n_outcome}\n",
                f"Number of diversion units: {self.n_diversion}\n",
            )
        else:
            num_units = f"Total number of units: {self.N}\n"
        tr_grid = f"Exposure grids: {self.treatment_grid}\n"
        if self.grid_treatment_gps is not None:
            nonzero_cts = np.count_nonzero(self.grid_treatment_gps, axis=1)
        gps_summary = (
            f"Number of units with nonzero GPS in each grid level: {nonzero_cts}\n"
        )
        # res = ("================== DataSet Object ==================\n"
        #     + "\n------------------ Data summary ------------------\n"
        #     + num_units
        #     + den_est
        #     + tr_grid
        #     + gps_summary
        #     f"Dataset Type: {str(type(self.data))}")
        # print(res)
        # return res

    def estimate_gps(
        self,
        exposure_vector: np.ndarray,
        grid_treatment: np.ndarray,
        learner: list,
    ):
        """
        Estimate generalized propensity score for
        1) R_hat(i) the GPS for the given observed treatment
        2) R_hat(i, t) with t in a grid of treatment levels

        Parameters:
            exposure_vector: (N, ) shape exposure vector
            grid_treatment: a grid of treatment levels e.g. [0,0.1,0.2,...,1]
            learner: estimator for GPS using histograms

        Returns:
            exposure_one_gps: (N, ) GPS estimates R_hat(i)
            grid_treatment_gps: (len(grid_treatment), N) GPS R_hat(i, t) with t in a grid of treatment levels
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

    def _compute_treatment_grid(self, n_grids):
        self.n_grids = n_grids
        self.treatment_grid = np.linspace(
            self.lower_grid_constraint, self.upper_grid_constraint, n_grids
        )

    def prepare_data(
        self,
        t_assignment: np.ndarray,
        t_assignment_sim: np.ndarray,
        n_grids: Union[int, None],
        treatment_grid: Union[np.ndarray, None],
        n_bins: int,
        density_estimator,
    ):
        """
        Prepare data for imputation-regression estimator

        Parameters:
        ----------

        t_assignment: (N, ) shape treatment assignment

        t_assignment_sim: (R, N) shape simulated treatment assignments

        n_grids: number of grids to evaluate estimated values.
        For instance, if the number 6 is selected, this means the algorithm will only take
        the 6 exposure levels at approximately 0, 0.2, 0.4, 0.6, 0.8, 1.0 to estimate the causal dose response curve.
        Higher value here means the final curve will be more finely estimated,
        but also increases computation time. Default is usually a reasonable number.

        treatment_grid: a treatment grid specified by user. must be None if n_grids is not None

        n_bins: number of bins to learn the gps using histograms.
        For example, when n_bins = 5, we create [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8) and [0.8, 1.0] bins
        and use the histograms to estimate probability that we observe a particular exposure level in each bin.
        Higher values mean the gps estimated more granualy.

        Returns:
        ----------

            self
        """
        if not isinstance(t_assignment, np.ndarray):
            raise TypeError(
                f"t_assignment must be an numpy array",
                f"but is of {type(t_assignment)}",
            )
        if n_grids and not isinstance(n_grids, int):
            raise TypeError(
                f"n_grids parameter must be an integer or None, "
                f"but found type {type(n_grids)}."
            )

        if isinstance(density_estimator, ReflectiveLearner):
            raise NotImplementedError

        if n_grids is not None and treatment_grid is not None:
            raise ValueError(
                "You must specify either n_grids or treatment_grid in argument, not both."
            )

        if not isinstance(n_bins, int):
            raise TypeError(
                f"n_bins parameter must be an integer, "
                f"but found type {type(n_bins)}."
            )

        if treatment_grid is None:
            # compute treatment_grid
            self._compute_treatment_grid(n_grids)
        else:
            self.treatment_grid = treatment_grid

        adj_matrix = self.data.adj_matrix.toarray()
        # compute exposure vector
        dist_tr = self.make_prob_dist(adj_matrix, t_assignment)
        if self._is_bipartite_data:
            dist_tr = dist_tr[: self.n_outcome]

        # compute simulated exposure vectors
        dist_sim = self.make_prob_dist(adj_matrix, t_assignment_sim)
        if self._is_bipartite_data:
            dist_sim = dist_sim[:, : self.n_outcome]

        # fit exposure density estimators on each outcome unit using simulated exposure vectors
        # dist_sim[:, i] R x 1

        density_estimator_list = [
            density_estimator(n_bins).fit(X=dist_sim[:, i]) for i in range(0, self.N)
        ]

        # estimate GPS for R_hat(i) and R_hat(i,t)
        exposure_gps, grid_treatment_gps = self.estimate_gps(
            exposure_vector=dist_tr,
            grid_treatment=self.treatment_grid,
            learner=density_estimator_list,
        )

        self.exposure_gps = exposure_gps
        self.grid_treatment_gps = grid_treatment_gps
        self.tr_vec = t_assignment
        self.tr_sim = t_assignment_sim
        self.exp_vec = dist_tr
        self.n_bins = n_bins
        self.density_estimator = density_estimator
        return self


class DesignDataLoader(DataLoader):

    def __init__(
        self,
        data: Dataset,
        lower_grid_constraint=0.0,
        upper_grid_constraint=1.0,
    ):
        super().__init__(data)
        self.tr_vec = None
        self.tr_sim = None
        self.exp_vec = None
        self.exposure_probs = None
        self.obs_exposure = None
        self.treatment_grid = None
        self.n_grids = None
        self.lower_grid_constraint = lower_grid_constraint
        self.upper_grid_constraint = upper_grid_constraint

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

    def make_obs_exposure(self, exposure_vec):
        """
        Args:
            exposure_vec: (N, ) exposure vector induced by a particular treatment vector

        Returns:
            bin_membership: true/false matrix of shape (N, K), each row represents a unit's status
            in K exposure levels (bins)
        """
        bin_edges = np.linspace(
            self.lower_grid_constraint, self.upper_grid_constraint, self.n_grids
        )
        bin_edges[-1] = 1.01
        # Create a boolean array indicating whether each element falls into each bin (1000, 10)
        bin_membership = (exposure_vec[:, np.newaxis] >= bin_edges[:-1]) & (
            exposure_vec[:, np.newaxis] < bin_edges[1:]
        )
        assert (
            bin_membership.shape[1] == self.n_grids - 1
        ), "number of bins = number of grids - 1"

        return bin_membership

    def make_exposure_prob(self, tr_sim_vec: np.ndarray, n_grids: int) -> tuple:
        """create a tuple of dictionaries of I_exposure, prob_exposure_k_k, prob_exposure_k_j:
        Args:
            tr_sim_vec: (R, N) simulated treatment assignment vector
            n_grids: K

        Returns:
            a tuple of:

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
        N = self.n_outcome if self._is_bipartite_data else self.n_units
        K = n_grids - 1
        R = tr_sim_vec.shape[0]

        adj_matrix = self.data.adj_matrix.toarray()
        # dist_tr = self.make_prob_dist(adj_matrix, tr_vec)
        # if self._is_bipartite_data:
        #     dist_tr = dist_tr[: self.n_outcome]

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

    def make_prob_exposure_cond(self, prob_exposure_k_k: Dict[str, Dict[str, Any]]):
        """

        Args:
            prob_exposure_k_k: a dict of K symmetric N*N numeric matrices
        Returns:
            prob_exposure_cond: K x N matrix representing estimated π_i (d_k),
            each row is [π_1(d_k), π_2(d_k),...π_N(d_k)]

        """

        K = self.n_grids - 1
        N = self.n_outcome if self._is_bipartite_data else self.n_units

        prob_exposure_cond = np.zeros((K, N))
        for j, exposure_name in enumerate(prob_exposure_k_k.keys()):
            prob_exposure_cond[j, :] = np.diag(prob_exposure_k_k[exposure_name])

        return prob_exposure_cond

    def _compute_treatment_grid(self, n_grids):
        self.n_grids = n_grids
        self.treatment_grid = np.linspace(
            self.lower_grid_constraint, self.upper_grid_constraint, n_grids
        )

    def prepare_data(self, t_assignment, t_assignment_sim, n_grids):
        """
        prepare data for design-based estimator
        """
        self._compute_treatment_grid(n_grids)

        # avoid recomputing exposure probs in repeated calls
        if self.exposure_probs is None:
            # compute exposure probabilities dictionaries
            I_exposure, prob_exposure_k_k, prob_exposure_k_l = self.make_exposure_prob(
                t_assignment_sim, n_grids
            )

            # compute generalized propensity score π_i(d_k)
            obs_prob_exposure_individual_kk = self.make_prob_exposure_cond(
                prob_exposure_k_k
            )

            self.exposure_probs = {
                "I_exposure": I_exposure,
                "prob_exposure_k_k": prob_exposure_k_k,
                "prob_exposure_k_l": prob_exposure_k_l,
                "obs_prob_exposure_individual_kk": obs_prob_exposure_individual_kk,
            }

        # compute exposure vector induced by treatment vector
        exp_vec = self.make_prob_dist(self.data.adj_matrix.toarray(), t_assignment)
        if self._is_bipartite_data:
            exp_vec = exp_vec[: self.n_outcome]

        # compute exposure in k conditions indicator matrix
        obs_exposure = self.make_obs_exposure(exp_vec)

        self.tr_vec = t_assignment
        self.tr_sim = t_assignment_sim
        self.exp_vec = exp_vec

        self.obs_exposure = obs_exposure

        return self
