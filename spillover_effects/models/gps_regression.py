import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    cross_val_score,
)


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, dataloader, Y, reg_model) -> None:
        self.dataloader = dataloader
        self.exposure_vec = self.dataloader.exp_vec
        self.Y = Y
        self.grid = self.dataloader.treatment_grid
        self.reg_model = reg_model
        self._n_units = len(self.exposure_vec)
        self._n_grids = self.dataloader.n_grids

    @property
    def n_units(self):
        return self._n_units

    @property
    def n_grids(self):
        return self._n_grids

    @abstractmethod
    def _prepare_X_train(self):
        pass

    @abstractmethod
    def _prepare_X_predict(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict_grid(self) -> np.ndarray:
        pass


class LinearModel(BaseModel):
    def __init__(
        self,
        dataloader,
        Y,
        reg_model,
        param_func,
        direct_treatment_vec=None,
    ):
        super().__init__(dataloader, Y, reg_model)
        self.param_func = param_func
        self.direct_treatment_vec = direct_treatment_vec

    def _prepare_X_train(self):
        X_train = self.param_func(self.exposure_vec)

        return X_train

    def _prepare_X_predict(self, e, t=None):
        """
        t: optional, direct treatment level,
        e: exposure level
        """
        if t:
            return self.param_func(e, t)
        else:
            return self.param_func(e)

    def fit(self):
        """fit a regression model"""
        X_train = self._prepare_X_train()
        self.reg_model.fit(X_train, self.Y)
        return self

    def _predict_grid_bipartite(self, grid) -> pd.DataFrame:
        """
        make inference on exposure level grids,
        we can specify any grid because the regression does not include gps
        """

        yt_grid = [0] * (len(grid))

        for i, val in enumerate(grid):
            X_predict = self.param_func(val)
            yt_grid[i] = self.reg_model.predict(X_predict).item()

        return pd.DataFrame({"grid": grid, "yt_grid_hat": yt_grid})

    def _predict_grid(self, grid: np.ndarray) -> pd.DataFrame:
        treatment, control = 1, 0
        yt_grid_1 = [0] * (len(grid))
        yt_grid_0 = [0] * (len(grid))

        for i, val in enumerate(grid):
            # e_predict = e / (len(grid) - 1)
            X_predict_0 = self.param_func(treatment, val)
            X_predict_1 = self.param_func(control, val)
            yt_grid_0[i] = self.reg_model.predict(X_predict_0).item()
            yt_grid_1[i] = self.reg_model.predict(X_predict_1).item()

        return pd.DataFrame(
            {"grid": grid, "yt_grid_hat_0": yt_grid_0, "yt_grid_hat_1": yt_grid_1}
        )

    def predict_grid(self) -> pd.DataFrame:
        if self.direct_treatment_vec:
            return self._predict_grid(self.grid)
        else:
            return self._predict_grid_bipartite(self.grid)


class HIModel(BaseModel):
    """Hirano-Imbens (2004) parametric imputation model"""

    def __init__(
        self,
        dataloader,
        Y,
        reg_model,
        param_func,
        direct_treatment_vec=None,
    ):
        super().__init__(dataloader, Y, reg_model)
        self.exposure_vec_gps = self.dataloader.exposure_gps
        self.grid_treatment_gps = self.dataloader.grid_treatment_gps
        self.param_func = param_func
        self.direct_treatment_vec = direct_treatment_vec

    def _prepare_X_train(self):
        """
        collect f(t, e, gps) or f(e, gps) as training data for first stage regression model
        """
        if self.direct_treatment_vec:
            X_train = self.param_func(
                self.direct_treatment_vec, self.exposure_vec, self.exposure_vec_gps
            )
        else:
            X_train = self.param_func(self.exposure_vec, self.exposure_vec_gps)
        return X_train

    def fit(self):
        """fit first stage regression model"""
        X_train = self._prepare_X_train()
        self.reg_model.fit(X_train, self.Y)
        return self

    def _predict_grid_bipartite(self) -> pd.DataFrame:
        """
        Given the estimated parameter in the second stage and the gps at grid exposure levels,
        we estimate the average potential outcome at treatment level t.
        """

        grid = self.grid
        yt_grid = [0] * (len(grid))
        for i, e_level in enumerate(grid):
            e_predict = np.full((self.n_units,), e_level)
            R_predict = self.grid_treatment_gps[i, :]  # 21*500
            X_predict = self.param_func(e_predict, R_predict)
            yt_grid[i] = np.mean(self.reg_model.predict(X_predict))

        return pd.DataFrame({"grid": grid, "yt_grid_hat": yt_grid})

    def _prepare_X_predict(self):
        raise NotImplementedError

    def _predict_grid(self) -> pd.DataFrame:
        grid = self.grid

        yt_grid_0 = [0] * (len(grid))
        yt_grid_1 = [0] * (len(grid))
        for i, e_level in enumerate(grid):
            e_predict = np.full((self.n_units,), e_level)
            R_predict = self.grid_treatment_gps[i, :]
            t_ind = np.ones_like(e_predict)
            c_ind = np.zeros_like(e_predict)
            X_predict_1 = self.param_func(t_ind, e_predict, R_predict)
            X_predict_0 = self.param_func(c_ind, e_predict, R_predict)
            yt_grid_0[i] = np.mean(self.reg_model.predict(X_predict_0))
            yt_grid_1[i] = np.mean(self.reg_model.predict(X_predict_1))

        return pd.DataFrame(
            {"grid": grid, "yt_grid_hat_0": yt_grid_0, "yt_grid_hat_1": yt_grid_1}
        )

    def predict_grid(self) -> pd.DataFrame:
        if self.direct_treatment_vec:
            return self._predict_grid()
        else:
            return self._predict_grid_bipartite()
