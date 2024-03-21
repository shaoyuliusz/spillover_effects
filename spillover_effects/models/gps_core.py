"""
Defines the Generalized Prospensity Score (GPS) Core model class
"""

import contextlib
import io

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype
from pygam import LinearGAM, LogisticGAM, s
from scipy.stats import gamma, norm
import statsmodels.api as sm
from statsmodels.genmod.families.links import inverse_power as Inverse_Power
from statsmodels.tools.tools import add_constant

from utils.dataloader import ImputeDataLoader
from .gps_base import Core


class GPS_Core(Core):
    """
     In a multi-stage approach, this computes the generalized propensity score (GPS) function,
     and uses this in a generalized additive model (GAM) to correct treatment prediction of
     the outcome variable. Assumes continuous treatment, but the outcome variable may be
     continuous or binary.

     WARNING:

     -This algorithm assumes you've already performed the necessary transformations to
     categorical covariates (i.e. these variables are already one-hot encoded and
     one of the categories is excluded for each set of dummy variables).

     -Please take care to ensure that the "ignorability" assumption is met (i.e.
     all strong confounders are captured in your covariates and there is no
     informative censoring), otherwise your results will be biased, sometimes strongly so.

     Parameters
     ----------

    spline_order: int, optional (default = 3)
         Order of the splines to use fitting the final GAM.
         Must be integer >= 1. Default value creates cubic splines.

     n_splines: int, optional (default = 30)
         Number of splines to use for the treatment and GPS in the final GAM.
         Must be integer >= 2. Must be non-negative.

     lambda_: int or float, optional (default = 0.5)
         Strength of smoothing penalty. Must be a positive float.
         Larger values enforce stronger smoothing.

     max_iter: int, optional (default = 100)
         Maximum number of iterations allowed for the maximum likelihood algo to converge.

     random_seed: int, optional (default = None)
         Sets the random seed.

     verbose: bool, optional (default = False)
         Determines whether the user will get verbose status updates.


     Attributes
     ----------

     grid_values: array of shape (treatment_grid_num, )
         The gridded values of the treatment variable. Equally spaced.

     best_gps_family: str
         If no gps_family is specified and the algorithm chooses the best glm family, this is
         the name of the family that was chosen.

     gps_deviance: float
         The GPS model deviance

     gps: array of shape (number of observations, )
         The GPS for each observation

     gam_results: `pygam.LinearGAM` class
         trained model of `LinearGAM` class, from pyGAM library


     Methods
     ----------
     fit: (self, T, X, y)
         Fits the causal dose-response model.

     calculate_CDRC: (self, ci)
         Calculates the CDRC (and confidence interval) from trained model.

     print_gam_summary: (self)
         Prints pyGAM text summary of GAM predicting outcome from the treatment and the GPS.

     References
     ----------

     Galagate, D. Causal Inference with a Continuous Treatment and Outcome: Alternative
     Estimators for Parametric Dose-Response function with Applications. PhD thesis, 2016.

     Moodie E and Stephens DA. Estimation of dose–response functions for
     longitudinal data using the generalised propensity score. In: Statistical Methods in
     Medical Research 21(2), 2010, pp.149–166.

     Hirano K and Imbens GW. The propensity score with continuous treatments.
     In: Gelman A and Meng XL (eds) Applied bayesian modeling and causal inference
     from incomplete-data perspectives. Oxford, UK: Wiley, 2004, pp.73–84.
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
        self.dataloader = dataloader
        self.spline_order = spline_order
        self.n_splines = n_splines
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.verbose = verbose

        self.grid_values = self.dataloader.treatment_grid
        self.treatment_grid_num = self.dataloader.n_grids
        if self.dataloader._is_bipartite_data:
            self.T = self.dataloader.tr_vec[: self.dataloader.n_outcome]
        else:
            self.T = self.dataloader.tr_vec

    def _validate_init_params(self):
        """
        Checks that the params used when instantiating GPS model are formatted correctly
        """
        # Checks for spline_order
        if not isinstance(self.spline_order, int):
            raise TypeError(
                f"spline_order parameter must be an integer, "
                f"but found type {type(self.spline_order)}"
            )

        if (isinstance(self.spline_order, int)) and self.spline_order < 1:
            raise ValueError(
                f"spline_order parameter should be >= 1, but found {self.spline_order}"
            )

        if (isinstance(self.spline_order, int)) and self.spline_order >= 30:
            raise ValueError("spline_order parameter is too high!")

        # Checks for n_splines
        if not isinstance(self.n_splines, int):
            raise TypeError(
                f"n_splines parameter must be an integer, but found type {type(self.n_splines)}"
            )

        if (isinstance(self.n_splines, int)) and self.n_splines < 2:
            raise ValueError(
                f"n_splines parameter should be >= 2, but found {self.n_splines}"
            )

        if (isinstance(self.n_splines, int)) and self.n_splines >= 100:
            raise ValueError("n_splines parameter is too high!")

        # Checks for lambda_
        if not isinstance(self.lambda_, (int, float)):
            raise TypeError(
                f"lambda_ parameter must be an int or float, but found type {type(self.lambda_)}"
            )

        if (isinstance(self.lambda_, (int, float))) and self.lambda_ <= 0:
            raise ValueError(
                f"lambda_ parameter should be > 0, but found {self.lambda_}"
            )

        if (isinstance(self.lambda_, (int, float))) and self.lambda_ >= 1000:
            raise ValueError("lambda_ parameter is too high!")

        # Checks for max_iter
        if not isinstance(self.max_iter, int):
            raise TypeError(
                f"max_iter parameter must be an int, but found type {type(self.max_iter)}"
            )

        if (isinstance(self.max_iter, int)) and self.max_iter <= 10:
            raise ValueError(
                "max_iter parameter is too low! Results won't be reliable!"
            )

        if (isinstance(self.max_iter, int)) and self.max_iter >= 1e6:
            raise ValueError("max_iter parameter is unnecessarily high!")

        # Checks for random_seed
        if not isinstance(self.random_seed, (int, type(None))):
            raise TypeError(
                f"random_seed parameter must be an int, but found type {type(self.random_seed)}"
            )

        if (isinstance(self.random_seed, int)) and self.random_seed < 0:
            raise ValueError("random_seed parameter must be > 0")

        # Checks for verbose
        if not isinstance(self.verbose, bool):
            raise TypeError(
                f"verbose parameter must be a boolean type, but found type {type(self.verbose)}"
            )

    def _validate_fit_data(self):
        """Verifies that T, X, and y are formatted the right way"""
        # Checks for T column
        if not is_float_dtype(self.T):
            raise TypeError("Treatment data must be of type float")

        # # Make sure all X columns are float or int
        # if isinstance(self.X, pd.Series):
        #     if not is_numeric_dtype(self.X):
        #         raise TypeError(
        #             "All covariate (X) columns must be int or float type (i.e. must be numeric)"
        #         )

        # elif isinstance(self.X, pd.DataFrame):
        #     for column in self.X:
        #         if not is_numeric_dtype(self.X[column]):
        #             raise TypeError(
        #                 "All covariate (X) columns must be int or float type "
        #                 "(i.e. must be numeric)"
        #             )

        # Checks for Y column
        if not (is_float_dtype(self.y) or is_integer_dtype(self.y)):
            raise TypeError("Outcome data must be of type float or integer")

        if is_integer_dtype(self.y) and (
            not np.array_equal(np.sort(self.y.unique()), np.array([0, 1]))
        ):
            raise TypeError(
                "If your outcome data is of type integer (binary outcome),"
                "it should only contain 1's and 0's."
            )

    def fit(self, y):
        """Fits the GPS causal dose-response model. For now, this only accepts pandas columns.
        While the treatment variable must be continuous (or ordinal with many levels), the
        outcome variable may be continuous or binary. You *must* provide
        at least one covariate column.

        Parameters
        ----------
        T: array-like, shape (n_samples,)
            A continuous treatment variable.
        X: array-like, shape (n_samples, m_features)
            Covariates, where n_samples is the number of samples
            and m_features is the number of features. Features can be a mix of continuous
            and nominal/categorical variables.
        y: array-like, shape (n_samples,)
            Outcome variable. May be continuous or binary. If continuous, this must
            be a series of type `float`, if binary must be a series of type `integer`.

        Returns
        ----------
        self : object

        """
        self.rand_seed_wrapper(self.random_seed)

        # self.T = self.T.reset_index(drop=True, inplace=False)
        # self.X = X.reset_index(drop=True, inplace=False)
        # self.y = y.reset_index(drop=True, inplace=False)
        self.y = y
        # Determine what type of outcome variable we're working with
        if is_float_dtype(y):
            self.outcome_type = "continuous"
        elif is_integer_dtype(y):
            self.outcome_type = "binary"

        self.if_verbose_print(
            f"Determined the outcome variable is of type {self.outcome_type}..."
        )

        # Validate this input data
        self._validate_fit_data()

        # Create grid_values
        # self.grid_values =

        # Determine which GPS family to use
        # self._determine_gps_function()

        # Estimate the GPS
        # self.if_verbose_print("Saving GPS values...")

        # self.gps = self.gps_function(self.T)

        # Create GAM that predicts outcome from the treatment and GPS
        self.if_verbose_print("Fitting GAM using treatment and GPS...")

        # Save model results
        self.gam_results = self._fit_gam()

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.gam_results.summary()

        self._gam_summary_str = f.getvalue()

        self.if_verbose_print(
            "Calculating many CDRC estimates for each treatment grid value..."
        )

        # Loop over all grid values (`treatment_grid_num` in total)
        # and give GPS loading for each observation in the dataset
        # self.gps_at_grid = self._gps_values_at_grid()

    def calculate_CDRC(self, ci=0.95):
        """Using the results of the fitted model, this generates a dataframe of
        point estimates for the CDRC at each of the values of the
        treatment grid. Connecting these estimates will produce the overall
        estimated CDRC. Confidence interval is returned as well.

        Parameters
        ----------
        ci: float (default = 0.95)
            The desired confidence interval to produce. Default value is 0.95, corresponding
            to 95% confidence intervals. bounded (0, 1.0).

        Returns
        ----------
        dataframe: Pandas dataframe
            Contains treatment grid values, the CDRC point estimate at that value,
            and the associated lower and upper confidence interval bounds at that point.

        self: object

        """
        self.rand_seed_wrapper(self.random_seed)
        self._validate_calculate_CDRC_params(ci)

        self.if_verbose_print(
            """
            Generating predictions for each value of treatment grid,
            and averaging to get the CDRC..."""
        )

        # Create CDRC predictions from trained GAM
        # If working with a continuous outcome variable, use this path
        if self.outcome_type == "continuous":
            self._cdrc_preds = self._cdrc_predictions_continuous(ci)

            results = []

            for i in range(0, self.treatment_grid_num):
                temp_grid_value = self.grid_values[i]
                temp_point_estimate = self._cdrc_preds[:, i, 0].mean()
                mean_ci_width = (
                    self._cdrc_preds[:, i, 2].mean() - self._cdrc_preds[:, i, 1].mean()
                ) / 2
                temp_lower_bound = temp_point_estimate - mean_ci_width
                temp_upper_bound = temp_point_estimate + mean_ci_width
                results.append(
                    [
                        temp_grid_value,
                        temp_point_estimate,
                        temp_lower_bound,
                        temp_upper_bound,
                    ]
                )

            outcome_name = "Causal_Dose_Response"

        # If working with a binary outcome variable, use this path
        else:
            self._cdrc_preds = self._cdrc_predictions_binary(ci)

            # Capture the first prediction's mean log odds.
            # This will serve as a reference for calculating the odds ratios
            log_odds_reference = self._cdrc_preds[:, 0, 0].mean()

            results = []

            for i in range(0, self.dataloader.n_bins):
                temp_grid_value = self.grid_values[i]

                temp_log_odds_estimate = (
                    self._cdrc_preds[:, i, 0].mean() - log_odds_reference
                )
                temp_OR_estimate = np.exp(temp_log_odds_estimate)

                temp_lower_bound = np.exp(
                    temp_log_odds_estimate
                    - (self.calculate_z_score(ci) * self._cdrc_preds[:, i, 1].mean())
                )
                temp_upper_bound = np.exp(
                    temp_log_odds_estimate
                    + (self.calculate_z_score(ci) * self._cdrc_preds[:, i, 1].mean())
                )
                results.append(
                    [
                        temp_grid_value,
                        temp_OR_estimate,
                        temp_lower_bound,
                        temp_upper_bound,
                    ]
                )

            outcome_name = "Causal_Odds_Ratio"

        return pd.DataFrame(
            results, columns=["Treatment", outcome_name, "Lower_CI", "Upper_CI"]
        ).round(3)

    @staticmethod
    def _validate_calculate_CDRC_params(ci):
        """Validates the parameters given to `calculate_CDRC`"""

        if not isinstance(ci, float):
            raise TypeError(
                f"`ci` parameter must be an float, but found type {type(ci)}"
            )

        if isinstance(ci, float) and ((ci <= 0) or (ci >= 1.0)):
            raise ValueError("`ci` parameter should be between (0, 1)")

    def print_gam_summary(self):
        """Prints the GAM model summary (uses pyGAM's output)

        Parameters
        ----------
        None

        Returns
        ----------
        self: object
        """
        print(self._gam_summary_str)

    def _fit_gam(self):
        """Fits a GAM that predicts the outcome (continuous or binary) from the treatment and GPS"""

        X = np.column_stack((self.T, self.dataloader.exposure_gps))
        y = np.asarray(self.y)

        model_type_dict = {"continuous": LinearGAM, "binary": LogisticGAM}

        return model_type_dict[self.outcome_type](
            s(0, n_splines=self.n_splines, spline_order=self.spline_order)
            + s(1, n_splines=self.n_splines, spline_order=self.spline_order),
            max_iter=self.max_iter,
            lam=self.lambda_,
        ).fit(X, y)
