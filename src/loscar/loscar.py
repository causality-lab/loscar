"""
Loscar (Location-Scale RESIT)

Extends the RESIT (regression with subsequent independence test) algorithm [1]
to LSNMs and integrated in-context quantile regression based on TabPFN.

Adapted from the `lingam` package (https://github.com/cdt15/lingam), file lingam/resit.py.
Copyright (c) 2019 T.Ikeuchi, G.Haraoka, M.Ide, W.Kurebayashi, S.Shimizu. Licensed under 
the MIT License, see LICENSES/lingam-MIT.txt.

References
----------
.. [1] Jonas Peters, Joris M Mooij, Dominik Janzing, and Bernhard Sch ̈olkopf.
    Causal discovery with continuous additive noise models.
    Journal of Machine Learning Research, 15:2009-2053, 2014.
"""

import numpy as np
from sklearn.utils import check_array
from scipy.stats import norm

from .base import _BaseLiNGAM
from .hsic import hsic_test_gamma

from .loscar_bivariate import (
    _quantile_residual,
    _quantile_residual_integrated,
    make_tabpfn_regressor,
)
from .ifgls_spline import _fgls_residual


class Loscar(_BaseLiNGAM):
    """Loscar (Location-Scale Resit)

    The implementation extends the RESIT implementation provided in the
    LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam

    Note: The adjacency matrix is transposed compared to the original implementation
    to comply with other baselines.

    Parameters
    ----------
    model : TabPFN instance. If not provided, a new model will be instantiated.
    random_state : int, optional (default=None)
    alpha : float, optional (default=0.01)
        Alpha level for HSIC independence test when removing superfluous edges.
    scale_quantiles: (low, high) quantile levels used to estimate a
        local scale. Only used when ``regressor="tabpfn"`` and
        ``scale_method="quantile_range"``.
    device: str, optional (default="cpu")
    regressor : str, optional (default="tabpfn") alternatively "fgls"
    fgls_kwargs : dict, optional (default=None)
    scale_method : str, optional (default="quantile_range") or "gauss_legendre"
    gl_m : int, optional (default=10)
        Number of Gauss-Legendre quadrature nodes.
    """

    def __init__(
        self,
        model=None,
        random_state=None,
        alpha=0.01,
        scale_quantiles=(0.05, 0.95),
        scale_method="quantile_range",
        qr_mean=False,
        gl_m=10,
        device="cpu",
        regressor="tabpfn",
        fgls_kwargs=None,
    ):
        if regressor not in ("tabpfn", "fgls"):
            raise ValueError(f"Unknown regressor: {regressor!r}")
        self.regressor = regressor
        self.fgls_kwargs = fgls_kwargs or {}
        self.qr_mean = qr_mean

        # Check parameters
        if self.regressor == "tabpfn":
            if model is None:
                self.model = make_tabpfn_regressor(
                    tabpfn_kwargs={
                        "device": device,
                        "model_path": "tabpfn-v3-regressor-v3_default.ckpt",
                        "random_state": 0,
                    }
                )
            else:
                self.model = model
        else:
            self.model = None  # not used by the fgls regressor

        self.q_lo, self.q_hi = scale_quantiles
        self.z_lo, self.z_hi = norm.ppf(self.q_lo), norm.ppf(self.q_hi)

        if scale_method not in ("quantile_range", "gauss_legendre"):
            raise ValueError(f"Unknown scale_method: {scale_method!r}")
        self.scale_method = scale_method

        if gl_m <= 1:
            raise ValueError("gl_m must be an integer greater than 0.")
        self.gl_m = gl_m

        if alpha < 0.0:
            raise ValueError("alpha must be an float greater than 0.")

        super().__init__(random_state)
        self._alpha = alpha

    def fit(self, X):
        """Skeleton of the RESIT Algorithm"""
        # Check parameters
        X = check_array(X)
        n_features = X.shape[1]

        # Determine topological order
        pa, pi = self._estimate_order(X)

        # Remove superfluous edges
        pa = self._remove_edges(X, pa, pi)

        # Create adjacency matrix from parent-child relationship
        adjacency_matrix = np.zeros([n_features, n_features])
        for i, parents in pa.items():
            for p in parents:
                adjacency_matrix[p, i] = 1

        self._causal_order = pi
        self._adjacency_matrix = adjacency_matrix
        return self

    def _estimate_order(self, X):
        """Determine topological order"""
        S = np.arange(X.shape[1])
        pa = {}
        pi = []

        for _ in range(X.shape[1]):
            if len(S) == 1:
                k = S[0]
            else:
                hsic_stats = []
                for k in S:
                    # Get residuals form Xk on {Xi}
                    predictors = [i for i in S if i != k]
                    residual = self._get_residuals(
                        X[:, predictors],
                        X[:, k],
                    )
                    # Measure dependence between residuals and {Xi}
                    hsic_stat, _ = hsic_test_gamma(residual, X[:, predictors])
                    hsic_stats.append(hsic_stat)

                k = S[np.argmin(hsic_stats)]

            S = S[S != k]
            pa[k] = S.tolist()
            pi.insert(0, k)

        return pa, pi

    def _remove_edges(self, X, pa, pi):
        """Remove superfluous edges"""
        for k in range(1, X.shape[1]):
            parents = pa[pi[k]].copy()

            for l in parents:
                # Regress Xk on {Xi}
                predictors = [i for i in pa[pi[k]] if i != l]
                if len(predictors) >= 1:
                    residual = self._get_residuals(
                        X[:, predictors],
                        X[:, pi[k]],
                    )
                else:
                    residual = X[:, pi[k]]
                # Measure dependence between residuals and {Xi}
                _, hsic_p = hsic_test_gamma(residual, X[:, parents])
                # eliminate edge
                if hsic_p > self._alpha:
                    pa[pi[k]].remove(l)

        return pa

    def _get_residuals(self, features, target):
        "Implements different regression frameworks for the location-scale residual."
        if self.regressor == "fgls":
            residual, *_ = _fgls_residual(features, target, self.fgls_kwargs)
            return residual
        elif self.regressor != "tabpfn":
            raise ValueError(f"Unknown regressor: {self.regressor!r}")

        if self.scale_method == "quantile_range":
            residual, *_ = _quantile_residual(
                self.model,
                features,
                target,
                self.q_lo,
                self.q_hi,
                self.z_lo,
                self.z_hi,
                self.qr_mean,
            )
            return residual
        elif self.scale_method == "gauss_legendre":
            residual, *_ = _quantile_residual_integrated(
                self.model, features, target, m=self.gl_m
            )
            return residual
        else:
            raise ValueError(f"Unknown scale_method: {self.scale_method!r}")
