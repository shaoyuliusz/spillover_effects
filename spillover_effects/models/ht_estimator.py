import numpy as np
from utils.dataloader import DesignDataLoader


class DesignEstimator:
    """Base class for design based estimator"""

    def __init__(self, dataloader: DesignDataLoader, obs_outcome):
        self.dataloader = dataloader
        self.obs_outcome = obs_outcome
        self.N = (
            self.dataloader.n_outcome
            if self.dataloader._is_bipartite_data
            else self.dataloader.n_units
        )

    def fit(self):
        """
        estimates
        Args:
            obs_exposure: N x K indicator matrix whether units N are in exposure condition k,
            where K is the total number of exposure conditions
            obs_outcome: (N, ) a vector length N of outcome data.
            obs_prob_exposure: a list of 3 lists containing exposure probabilities

        Returns:
            a dictionary of estimated results
        """
        obs_exposure = self.dataloader.obs_exposure

        # create (K, N) matrix, where non-zero entries are observed outcomes in a particular exposure level k.
        obs_outcome_by_exposure = obs_exposure.T.dot(np.diag(self.obs_outcome))

        # When πi(dk) = 0 for some units, then design-based estimation of average potential outcomes and
        # causal effects must be restricted to the subset of units for which πi (dk ) > 0.

        # set entry to nan when we do not observe that particular level of exposure.
        obs_outcome_by_exposure[np.where(obs_exposure.T == 0)] = np.nan

        yT_ht = np.nansum(
            obs_outcome_by_exposure
            / self.dataloader.exposure_probs["obs_prob_exposure_individual_kk"],
            axis=1,
        )  # (K, ) vector, sum each row.
        # np.all(np.isnan(obs_outcome_by_exposure), axis=1)  # a (K, ) array indicating if exposure level k is all nan
        yT_ht[np.where(np.all(np.isnan(obs_outcome_by_exposure), axis=1) == True)] = (
            np.nan
        )

        self.obs_outcome_by_exposure = obs_outcome_by_exposure
        self.yT_ht = yT_ht
        return self

    def __str__(self):
        class_name = self.__class__.__name__
        header = f"================== {class_name} Object ==================\n"
        data_summary = self.yT_ht
        score_info = self.var_yT_ht_adjusted

    def var_yT_ht_adjusted(self):
        var = self._var_yT_ht_unadjusted(
            self.dataloader.obs_exposure,
            self.obs_outcome,
            self.dataloader.exposure_probs,
        )
        A2 = self._var_yT_ht_A2_adjustment(
            self.dataloader.obs_exposure,
            self.obs_outcome,
            self.dataloader.exposure_probs,
        )

        var_yT_ht = var + A2
        # Set NaN for rows where all values in obs_outcome_by_exposure are NA
        var_yT_ht[np.sum(~np.isnan(self.obs_outcome_by_exposure), axis=1) == 0] = np.nan
        self.var_yT_ht = var_yT_ht
        return self

    def _var_yT_ht_unadjusted(self, obs_exposure, obs_outcome, prob_exposure):
        num_columns = obs_exposure.shape[1]
        var_yT = np.full(shape=(num_columns,), fill_value=np.nan)

        for k in range(num_columns):
            pi_k = prob_exposure["prob_exposure_k_k"][f"t{k},t{k}"]
            ind_kk = np.diag(pi_k)
            cond_indicator = obs_exposure[:, k]

            mm = (
                np.outer(cond_indicator, cond_indicator)
                * ((pi_k - np.outer(ind_kk, ind_kk)) / pi_k)
                * np.outer(obs_outcome, obs_outcome)
                / np.outer(ind_kk, ind_kk)
            )
            mm[~np.isfinite(mm)] = 0

            second_part_sum = np.sum(mm)

            var_yT[k] = second_part_sum

        return var_yT

    def _var_yT_ht_A2_adjustment(
        self, obs_exposure, obs_outcome, prob_exposure
    ) -> np.ndarray:
        """
        compute the A2 adjustment term as in Proposition 5.1 in Aronow and Samii (2017)
        Args:
            obs_exposure:
            obs_outcome:
            prob_exposure:

        Returns:
            var_yT_A2: numpy array of (K, ) shape hat{A2_{d_k}}
        """

        num_columns = obs_exposure.shape[1]
        var_yT_A2 = np.full(shape=(num_columns,), fill_value=np.nan)

        for k in range(num_columns):
            pi_k = prob_exposure["prob_exposure_k_k"][f"t{k},t{k}"]
            ind_kk = np.diag(pi_k)
            cond_indicator = obs_exposure[:, k]

            m = cond_indicator * (obs_outcome**2) / (2 * ind_kk)
            A2_part_sum = np.sum(
                np.outer(m, m)
                * (pi_k == 0)
                * np.logical_not(np.eye(len(m), dtype=bool))
            )
            var_yT_A2[k] = A2_part_sum

        return var_yT_A2

    def tau_ht(self, d_k=1, d_j=0):
        """compute tau_ht(d_k, d_l)
        d_k and d_l must be between 0 and 1 indicating dosage levels, defaults to 1 and 0
        if float is passed, map it to closest dosage grid and compute.
        """
        if not isinstance(d_k, (float, int)):
            raise TypeError(
                f"d_k parameter must be an integer or float, "
                f"but found type {type(d_k)}"
            )
        if not isinstance(d_j, (float, int)):
            raise TypeError(
                f"d_l parameter must be an integer or float, "
                f"but found type {type(d_j)}"
            )
        if d_k > 1 or d_k < 0:
            raise ValueError(
                f"d_k parameter must be between 0 and 1, " f"but found {d_k}"
            )
        if d_j > 1 or d_j < 0:
            raise ValueError(
                f"d_k parameter must be between 0 and 1, " f"but found {d_j}"
            )

        tau_ht = (1 / self.N) * (self.yT_ht[-1] - self.yT_ht[0])

        self.tau_ht = tau_ht
        return self

    def var_tau_ht(self, d_k=1, d_j=0):
        """
        need to first compute var_yT_ht
        """
        # options = np.linspace(0, 1, self.dataloader.n_bins + 1)
        # d_k_cl_index = np.argmin(np.abs(options - d_k))
        # d_j_cl_index = np.argmin(np.abs(options - d_j))

        cov_yT_A = self._cov_yT_ht_adjusted(
            self.dataloader.obs_exposure,
            self.obs_outcome,
            self.dataloader.exposure_probs,
            k_to_include=[d_k],
            j_to_include=[d_j],
        )

        cov_yT_A = cov_yT_A[f"{d_k},{d_j}"]

        var_tau_ht = (1 / self.N**2) * (
            self.var_yT_ht[-1] + self.var_yT_ht[0] - 2 * cov_yT_A
        )
        self.var_tau_ht = var_tau_ht
        return self

    def _cov_yT_ht_adjusted(
        self, obs_exposure, obs_outcome, prob_exposure, k_to_include, j_to_include
    ):
        """
        k_to_include: a list of k dosage levels to include
        l_to_include: a list of l dosage levels to include

        Returns:
        dict
        """
        N, K = obs_exposure.shape
        # obs_exposure: (N,K)
        # num_combinations = (
        #     2 * (obs_exposure.shape[1] * (obs_exposure.shape[1] - 1)) // 2
        # )
        # cov_yT_A = np.full(shape=(num_combinations,), fill_value=np.nan)
        if not k_to_include:
            k_to_include = range(K)
        if not j_to_include:
            j_to_include = range(K)
        cov_yT_A = {}

        for k in k_to_include:
            for j in j_to_include:
                if k != j:
                    pi_k = prob_exposure["prob_exposure_k_k"][f"t{k},t{k}"]
                    ind_kk = np.diag(pi_k)
                    pi_j = prob_exposure["prob_exposure_k_k"][f"t{j},t{j}"]
                    ind_jj = np.diag(pi_j)
                    pi_k_j = prob_exposure["prob_exposure_k_l"][f"t{k},t{j}"]
                    cond_indicator_k = obs_exposure[:, k]
                    cond_indicator_j = obs_exposure[:, j]

                    mm = (
                        np.outer(cond_indicator_k, cond_indicator_j)
                        * ((pi_k_j - np.outer(ind_kk, ind_jj)) / pi_k_j)
                        * np.outer(obs_outcome, obs_outcome)
                        / np.outer(ind_kk, ind_jj)
                    )
                    mm[~np.isfinite(mm)] = 0
                    first_part_cov = np.sum(mm)

                    second_part_cov = 0
                    for m in range(len(cond_indicator_k)):
                        for n in range(len(cond_indicator_j)):
                            if pi_k_j[m, n] == 0:
                                second_part_cov_m_n = (
                                    cond_indicator_k[m]
                                    * obs_outcome[m] ** 2
                                    / (2 * ind_kk[m])
                                ) + (
                                    cond_indicator_j[n]
                                    * obs_outcome[n] ** 2
                                    / (2 * ind_jj[n])
                                )
                                second_part_cov += second_part_cov_m_n

                    cov_yT_A[f"{k},{j}"] = first_part_cov - second_part_cov

        return cov_yT_A


if __name__ == "__main__":
    P_TREATMENT = 0.5
    N_OUTCOME = 1000
    N_DIVERSION = 100
    N_RESAMPLE = 500
    N_BINS = 10
    import numpy as np
    from datapipes.make_graph import make_bipartite_graph, make_sq_lattice_graph
    from dataloader import DesignDataLoader
    from dataset import BipartiteDataset
    import networkx as nx
    from spillover_effects.models.ht_estimator import DesignEstimator

    bigraph = make_bipartite_graph(N_OUTCOME, N_DIVERSION)
    # bigraph = make_sq_lattice_graph(N=9)
    adj_matrix = nx.adjacency_matrix(bigraph).toarray()
    degrees = [bigraph.degree(node) for node in range(0, N_OUTCOME)]

    dt = BipartiteDataset(graph=bigraph)
    data = DesignDataLoader(
        data=dt,
        n_bins=N_BINS,
        p_treatment=P_TREATMENT,
        n_resample=N_RESAMPLE,
        num_workers=6,
    )
    data.prepare_data(random_seed_tr=None, random_seed_sim=None)
    # outcome
    mu_e_hetero = degrees * data.exp_vec
    # mu_e_const = np.mean(degrees) * dist_tr
    obs_outcome = mu_e_hetero
    # obs_outcome_grid = [np.sum(degrees) * num for num in np.linspace(0, 1, N_BINS + 1)]

    ht_est = DesignEstimator(data, obs_outcome)
    ht_est.fit().var_yT_ht_adjusted()
    ht_est.tau_ht().var_tau_ht()
