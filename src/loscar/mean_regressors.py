"""Mean regressors (sklearn-style ``fit``/``predict``) for use with ``resit.RESIT``."""
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import SplineTransformer


def _as_2d(X):
    X = np.asarray(X)
    return X.reshape(-1, 1) if X.ndim == 1 else X


class TabPFNMeanRegressor:
    """Wraps a `tabpfn.TabPFNRegressor` and predicts the conditional mean.

    Parameters
    ----------
    model : a `tabpfn.TabPFNRegressor` instance
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(_as_2d(X), y)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(_as_2d(X), output_type="mean"))


class SplineRegression:
    """Predicts the conditional mean via a ridge-penalized OLS fit on a spline basis
    (the mean-fit step of `ifgls_spline.ifgls`).

    Parameters
    ----------
    n_knots_mean : int, default=6
    ridge_alpha_mean : float, default=1e-4
    """

    def __init__(self, n_knots_mean=6, ridge_alpha_mean=1e-4):
        self.n_knots_mean = n_knots_mean
        self.ridge_alpha_mean = ridge_alpha_mean
        self._feature_map = None
        self._hat_beta = None

    def fit(self, X, y):
        self._feature_map = SplineTransformer(
            n_knots=self.n_knots_mean,
            degree=3,
            include_bias=False,
            extrapolation="linear",
        )
        Phi_X = self._feature_map.fit_transform(_as_2d(X))
        Phi = np.column_stack([np.ones(Phi_X.shape[0]), Phi_X])
        mean_ols = sm.OLS(y, Phi).fit_regularized(
            L1_wt=0.0, alpha=self.ridge_alpha_mean
        )
        self._hat_beta = mean_ols.params
        return self

    def predict(self, X):
        Phi_X = self._feature_map.transform(_as_2d(X))
        Phi = np.column_stack([np.ones(Phi_X.shape[0]), Phi_X])
        return Phi @ self._hat_beta
