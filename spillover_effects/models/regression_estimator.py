import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    cross_val_score,
)


def model_naive(treatment_vec, exposure_vec, Y_obs, grid):
    """
    Estimate E(Y|T, E) using a linear regression without GPS
    Yi = α + βTi + γei + εi
    Args:
        treatment_vector: (N,1) vector for direct treatment assignment
        exposure_one: (N,1) the exposure level for each unit given the treatment vector
        Y_obs: observed outcomes
        grid:
    """
    assert len(treatment_vec.shape) == 1

    e = exposure_vec.reshape(-1)
    X_train = np.column_stack((treatment_vec, e))

    reg = LinearRegression().fit(X_train, Y_obs)

    yt_grid_1 = [0] * (len(grid))
    yt_grid_0 = [0] * (len(grid))

    for e in range(0, len(grid)):
        X_predict_0 = np.array((0, e / (len(grid) - 1))).reshape(1, -1)
        X_predict_1 = np.array((1, e / (len(grid) - 1))).reshape(1, -1)
        yt_grid_0[e] = reg.predict(X_predict_0).item()
        yt_grid_1[e] = reg.predict(X_predict_1).item()  # X_predict_0 shape (1, 2)

    return yt_grid_0, yt_grid_1


def model_random_forest(
    treatment_vec, exposure_vec, Y_obs, grid, exposure_vec_gps, grid_treatment_gps
):
    """
    Estimate E(Y|T, E) using a random forest model with/wo GPS
    """

    if len(treatment_vec.shape) == 2:
        treatment_vec = treatment_vec.reshape(-1)

    if exposure_vec_gps is not None:
        R_hat = np.array(exposure_vec_gps)

    e = exposure_vec.reshape(-1)
    X_train = np.column_stack((treatment_vec, e, R_hat))

    model = RandomForestRegressor()
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
    )
    grid_search.fit(X_train, Y_obs)
    best_rf = grid_search.best_estimator_

    yt_grid_1 = [0] * (len(grid))
    yt_grid_0 = [0] * (len(grid))

    N_UNITS = len(treatment_vec)

    for t in range(0, len(grid)):
        t_predict = np.zeros((N_UNITS,))
        t_predict.fill(t / (len(grid) - 1))
        R_predict = grid_treatment_gps[t, :]  # 21*500

        X_predict_1 = np.column_stack((np.ones_like(t_predict), t_predict, R_predict))
        X_predict_0 = np.column_stack((np.zeros_like(t_predict), t_predict, R_predict))
        yt_grid_1[t] = np.mean(best_rf.predict(X_predict_1))
        yt_grid_0[t] = np.mean(best_rf.predict(X_predict_0))

    return yt_grid_0, yt_grid_1


def model_interact(treatment_vec, exposure_vec, Y_obs, grid):
    """
    Estimate E(Y|T, E) using a linear regression interactive model without GPS
    Yi = α + βTi + γei + δ(ei × Ti) + u′

    Returns:
        yt_grid: estimated y(1, t) and y(0, t) at exposure level t.
    """
    if len(treatment_vec.shape) == 2:
        treatment_vec = treatment_vec.reshape(-1)
    t = exposure_vec
    X_train = np.column_stack((treatment_vec, t, treatment_vec * t))

    reg = LinearRegression().fit(X_train, Y_obs)

    yt_grid_1 = [0] * (len(grid))
    yt_grid_0 = [0] * (len(grid))
    nbins = len(grid) - 1
    for t in range(0, len(grid)):
        X_predict_1 = np.column_stack((1, t / nbins, 1 * t / nbins))
        X_predict_0 = np.column_stack((0, t / nbins, 0 * t / nbins))
        yt_grid_1[t] = reg.predict(X_predict_1).item()
        yt_grid_0[t] = reg.predict(X_predict_0).item()

    return yt_grid_0, yt_grid_1


def model_hi(
    treatment_vec, exposure_vec, Y_obs, grid, exposure_vec_gps, grid_treatment_gps
):
    """
    Hirano and Imbens (2004) polynomial regression
    Second step: Estimating E(Y|T, GPS) the conditional expectation of the outcome given the treatment and GPS
    """
    if len(treatment_vec.shape) == 2:
        treatment_vec = treatment_vec.reshape(-1)
    N_UNITS = len(treatment_vec)

    # after generating the GPS, use polynomials to approximate E(Y|T, p(X))
    R_hat = exposure_vec_gps
    e = exposure_vec

    # Stack [1, t,t^2, t^3, R, R^2, R^3, t*R] horizontally to form the matrix
    # print("R_hat: ", R_hat.shape)
    # print("t", t.shape)
    # print("treatment_vector", treatment_vector.shape)

    X = np.column_stack(
        (
            treatment_vec,
            e,
            R_hat,
            R_hat * e,
        )
    )

    reg = LinearRegression().fit(X, Y_obs)

    yt_grid_1 = [0] * (len(grid))
    yt_grid_0 = [0] * (len(grid))
    for t in range(0, len(grid)):
        t_predict = np.zeros((N_UNITS,))
        t_predict.fill(t / (len(grid) - 1))
        R_predict = grid_treatment_gps[t, :]  # 21*500

        X_predict_1 = np.column_stack(
            (
                np.ones_like(t_predict),
                t_predict,
                R_predict,
                t_predict * R_predict,
            )
        )

        X_predict_0 = np.column_stack(
            (
                np.zeros_like(t_predict),
                t_predict,
                R_predict,
                t_predict * R_predict,
            )
        )
        # X_predict_0: (n_units, n_features) (500,9)
        yt_grid_1[t] = np.mean(reg.predict(X_predict_1))
        yt_grid_0[t] = np.mean(reg.predict(X_predict_0))

    return yt_grid_0, yt_grid_1


def model_naive_bipartite(exposure_vec, Y_obs, grid):
    """
    Estimate E(Y|T, E) using a linear regression without GPS
    Yi = α + βTi + γei + εi
    Args:
        treatment_vector: (N,1) vector for direct treatment assignment
        exposure_one: (N,1) the exposure level for each unit given the treatment vector
        Y_obs: observed outcomes
        grid:
    """

    X_train = exposure_vec.reshape(-1, 1)

    reg = LinearRegression().fit(X_train, Y_obs)

    yt_grid = [0] * (len(grid))

    for e in range(0, len(grid)):
        X_predict = np.array((e / (len(grid) - 1))).reshape(1, -1)
        yt_grid[e] = reg.predict(X_predict).item()

    return yt_grid


def model_hi_bipartite(exposure_vec, Y_obs, grid, exposure_vec_gps, grid_treatment_gps):
    """
    Hirano and Imbens (2004) polynomial regression
    Second step: Estimating E(Y|T, GPS) the conditional expectation of the outcome given the treatment and GPS
    """

    N_UNITS = len(exposure_vec)

    # after generating the GPS, use polynomials to approximate E(Y|T, p(X))
    R_hat = exposure_vec_gps
    e = exposure_vec

    # Stack [1, t,t^2, t^3, R, R^2, R^3, t*R] horizontally to form the matrix
    # print("R_hat: ", R_hat.shape)
    # print("t", t.shape)
    # print("treatment_vector", treatment_vector.shape)

    X = np.column_stack(
        (
            e,
            e**2,
            R_hat,
            R_hat**2,
            R_hat * e,
        )
    )

    reg = LinearRegression().fit(X, Y_obs)

    yt_grid = [0] * (len(grid))
    for t in range(0, len(grid)):
        t_predict = np.zeros((N_UNITS,))
        t_predict.fill(t / (len(grid) - 1))
        R_predict = grid_treatment_gps[t, :]  # 21*500

        X_predict = np.column_stack(
            (
                t_predict,
                t_predict**2,
                R_predict,
                R_predict**2,
                t_predict * R_predict,
            )
        )

        # X_predict_0: (n_units, n_features) (500,9)
        yt_grid[t] = np.mean(reg.predict(X_predict))

    return yt_grid


def model_randomforest_bipartite(
    exposure_vec, Y_obs, grid, exposure_vec_gps, grid_treatment_gps
):
    """
    Estimate E(Y|T, E) using a random forest model with/wo GPS
    """

    if exposure_vec_gps is not None:
        R_hat = np.array(exposure_vec_gps)

    e = exposure_vec.reshape(-1)
    X_train = np.column_stack((e, R_hat))

    model = GradientBoostingRegressor()
    param_grid = {
        "n_estimators": [50, 100, 200],
        # "max_depth": [None, 10, 20],
        # "min_samples_split": [2, 5, 10],
        # "min_samples_leaf": [1, 2, 4],
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
    )
    grid_search.fit(X_train, Y_obs)
    best_rf = grid_search.best_estimator_

    yt_grid = [0] * (len(grid))

    N_UNITS = len(exposure_vec)

    for t in range(0, len(grid)):
        t_predict = np.zeros((N_UNITS,))
        t_predict.fill(t / (len(grid) - 1))
        R_predict = grid_treatment_gps[t, :]  # 21*500

        X_predict = np.column_stack((t_predict, R_predict))
        yt_grid[t] = np.mean(best_rf.predict(X_predict))

    return yt_grid
