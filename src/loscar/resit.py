"""
RESIT (regression with subsequent independence test) algorithm [1].

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

from .base import _BaseLiNGAM
from .hsic import hsic_test_gamma


class RESIT(_BaseLiNGAM):
    """RESIT (regression with subsequent independence test) algorithm.

    Note: The adjacency matrix is transposed compared to the original implementation
    to comply with other baselines.
    """

    def __init__(self, regressor, random_state=None, alpha=0.01):
        """Construct a RESIT model.

        Parameters
        ----------
        regressor : regressor object implementing 'fit' and 'predict', e.g.
            `TabPFNMeanRegressor` or `SplineRegression` from `mean_regressors`.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        alpha : float, optional (default=0.01)
            Alpha level for HSIC independence test when removing superfluous edges.
        """
        if regressor is None:
            raise ValueError("Specify regression model in 'regressor'.")
        if not (hasattr(regressor, "fit") and hasattr(regressor, "predict")):
            raise ValueError("'regressor' has no fit or predict method.")
        if alpha < 0.0:
            raise ValueError("alpha must be an float greater than 0.")

        super().__init__(random_state)
        self._alpha = alpha
        self._reg = regressor

    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X)
        n_features = X.shape[1]

        pa, pi = self._estimate_order(X)
        pa = self._remove_edges(X, pa, pi)

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
                    predictors = [i for i in S if i != k]
                    self._reg.fit(X[:, predictors], X[:, k])
                    residual = X[:, k] - self._reg.predict(X[:, predictors])

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
                predictors = [i for i in pa[pi[k]] if i != l]
                if len(predictors) >= 1:
                    self._reg.fit(X[:, predictors], X[:, pi[k]])
                    residual = X[:, pi[k]] - self._reg.predict(X[:, predictors])
                else:
                    residual = X[:, pi[k]]
                _, hsic_p = hsic_test_gamma(residual, X[:, parents])
                if hsic_p > self._alpha:
                    pa[pi[k]].remove(l)

        return pa
