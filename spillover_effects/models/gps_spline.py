"""
Defines the Generalized Prospensity Score (GPS) regressor model class
"""

from pprint import pprint
import numpy as np
from gps_core import GPS_Core
from ..utils.dataloader import ImputeDataLoader


class GPS_SplineRegressor(GPS_Core):
    """
    A GPS tool that handles continuous outcomes. Inherits the GPS_core
    base class. See that base class code its docstring for more details.

    Methods
    ----------

    point_estimate: (self, T)
        Calculates point estimate within the CDRC given treatment values.
        Can only be used when outcome is continuous.

    point_estimate_interval: (self, T, ci)
        Calculates the prediction confidence interval associated with a point estimate
        within the CDRC given treatment values. Can only be used when outcome is continuous.

    """

    def __init__(
        self,
        dataloader: ImputeDataLoader,
        spline_order=3,
        n_splines=30,
        lambda_=0.5,
        max_iter=100,
        random_seed=None,
        verbose=False,
    ):
        super().__init__(
            dataloader, spline_order, n_splines, lambda_, max_iter, random_seed, verbose
        )
        # self.grid_value = self.dataloader.treatment_grid
        # self.treatment_grid_num = self.dataloader.n_bins
        # Validate the params
        # self._validate_init_params()
        self.rand_seed_wrapper()

        self.if_verbose_print("Using the following params for GPS model:")
        if self.verbose:
            pprint(self.get_params(), indent=4)

    def _cdrc_predictions_continuous(self, ci):
        """Returns the predictions of CDRC for each value of the treatment grid. Essentially,
        we're making predictions using the original treatment and gps_at_grid.
        To be used when the outcome of interest is continuous.
        """

        # To keep track of cdrc predictions, we create an empty 3d array of shape
        # (n_samples, treatment_grid_num, 3). The last dimension is of length 3 because
        # we are going to keep track of the point estimate of the prediction, as well as
        # the lower and upper bounds of the prediction interval
        cdrc_preds = np.zeros((len(self.T), self.treatment_grid_num, 3), dtype=float)

        # Loop through each of the grid values, predict point estimate and get prediction interval
        for i in range(0, self.treatment_grid_num):  # (K)
            temp_T = np.repeat(self.grid_values[i], repeats=len(self.T))  # (N, )
            temp_gps = self.dataloader.grid_treatment_gps[i, :]  # (K,N)
            temp_cdrc_preds = self.gam_results.predict(
                np.column_stack((temp_T, temp_gps))
            )
            temp_cdrc_interval = self.gam_results.confidence_intervals(
                np.column_stack((temp_T, temp_gps)), width=ci
            )
            temp_cdrc_lower_bound = temp_cdrc_interval[:, 0]
            temp_cdrc_upper_bound = temp_cdrc_interval[:, 1]
            cdrc_preds[:, i, 0] = temp_cdrc_preds
            cdrc_preds[:, i, 1] = temp_cdrc_lower_bound
            cdrc_preds[:, i, 2] = temp_cdrc_upper_bound

        return np.round(cdrc_preds, 3)

    # def point_estimate(self, T):
    #     """Calculates point estimate within the CDRC given treatment values.
    #     Can only be used when outcome is continuous. Can be estimate for a single
    #     data point or can be run in batch for many observations. Extrapolation will produce
    #     untrustworthy results; the provided treatment should be within
    #     the range of the training data.

    #     Parameters
    #     ----------
    #     T: Numpy array, shape (n_samples,)
    #         A continuous treatment variable.

    #     Returns
    #     ----------
    #     array: Numpy array
    #         Contains a set of CDRC point estimates

    #     """
    #     if self.outcome_type != "continuous":
    #         raise TypeError("Your outcome must be continuous to use this function!")

    #     return np.apply_along_axis(self._create_point_estimate, 0, T.reshape(1, -1))

    # def _create_point_estimate(self, T):
    #     """Takes a single treatment value and produces a single point estimate
    #     in the case of a continuous outcome.
    #     """
    #     return self.gam_results.predict(
    #         np.array([T, self.gps_function(T).mean()]).reshape(1, -1)
    #     )

    # def point_estimate_interval(self, T, ci=0.95):
    #     """Calculates the prediction confidence interval associated with a point estimate
    #     within the CDRC given treatment values. Can only be used
    #     when outcome is continuous. Can be estimate for a single data point
    #     or can be run in batch for many observations. Extrapolation will produce
    #     untrustworthy results; the provided treatment should be within
    #     the range of the training data.

    #     Parameters
    #     ----------
    #     T: Numpy array, shape (n_samples,)
    #         A continuous treatment variable.
    #     ci: float (default = 0.95)
    #         The desired confidence interval to produce. Default value is 0.95, corresponding
    #         to 95% confidence intervals. bounded (0, 1.0).

    #     Returns
    #     ----------
    #     array: Numpy array
    #         Contains a set of CDRC prediction intervals ([lower bound, higher bound])

    #     """
    #     if self.outcome_type != "continuous":
    #         raise TypeError("Your outcome must be continuous to use this function!")

    #     return np.apply_along_axis(
    #         self._create_point_estimate_interval, 0, T.reshape(1, -1), width=ci
    #     ).T.reshape(-1, 2)

    # def _create_point_estimate_interval(self, T, width):
    #     """Takes a single treatment value and produces confidence interval
    #     associated with a point estimate in the case of a continuous outcome.
    #     """
    #     return self.gam_results.prediction_intervals(
    #         np.array([T, self.gps_function(T).mean()]).reshape(1, -1), width=width
    #     )
