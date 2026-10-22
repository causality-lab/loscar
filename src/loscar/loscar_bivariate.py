"""
Loscar

Implementation for Cause-Effect pairs. Contains further functinonality for
(multivariate) quantile regression used in other files.
"""

import numpy as np
from scipy.stats import norm
from tabpfn import TabPFNRegressor

from .hsic import hsic_test_gamma


def _quadrature_nodes(m):
    """Gauss-Legendre quadrature nodes/weights on (0, 1) for a Uniform(0,1) measure.

    Parameters
    ----------
    - m: number of quadrature nodes (integer)

    Returns
    -------
    - nodes: quantile levels in (0, 1) (1D NumPy array)
    - weights: quadrature weights, summing to 1 (1D NumPy array)
    """
    if m <= 1:
        return np.array([0.5]), np.array([1.0])
    nodes, weights = np.polynomial.legendre.leggauss(m)
    return 0.5 * (nodes + 1), 0.5 * weights


def make_tabpfn_regressor(tabpfn_kwargs=None, random_state=0, device="cpu"):
    """Build a TabPFNRegressor used for in-context quantile regression.

    Parameters
    ----------
    - tabpfn_kwargs: keyword arguments for `tabpfn.TabPFNRegressor`
      (dict, default=None -> {}). `device` defaults to "cpu".
    - random_state: default TabPFN random state if not set in
      `tabpfn_kwargs` (integer, default=0)

    Returns
    -------
    - a `tabpfn.TabPFNRegressor` instance
    """
    tabpfn_kwargs = dict(tabpfn_kwargs or {})
    tabpfn_kwargs.setdefault("device", device)
    tabpfn_kwargs.setdefault("random_state", random_state)
    tabpfn_kwargs.setdefault("model_path", "tabpfn-v3-regressor-v3_default.ckpt")
    return TabPFNRegressor(**tabpfn_kwargs)


def _fit_conditional_quantiles(model, cause, effect, quantile_levels):
    """Fit `model` on (cause -> effect) and predict in-sample quantiles.

    Parameters
    ----------
    - model: a `tabpfn.TabPFNRegressor` instance
    - cause: predictor variable (n,)
    - effect: target variable (n,)
    - quantile_levels: quantile levels to predict (1D array-like)

    Returns
    -------
    - n x len(quantile_levels) array of predicted conditional quantiles
    """
    cause = np.asarray(cause)
    if cause.ndim == 1:
        cause = cause.reshape(-1, 1)
    model.fit(cause, effect)
    preds = model.predict(
        cause,
        output_type="quantiles",
        quantiles=list(quantile_levels),
    )
    return np.column_stack(preds)


def _fit_conditional_quantiles_and_mean(model, cause, effect, quantile_levels):
    """Fit `model` on (cause -> effect) and predict in-sample quantiles and the mean.

    Parameters
    ----------
    - model: a `tabpfn.TabPFNRegressor` instance
    - cause: predictor variable(s) ((n,) or (n, p))
    - effect: target variable (n,)
    - quantile_levels: quantile levels to predict (1D array-like)

    Returns
    -------
    - mean: predicted conditional mean (n,)
    - quantiles: n x len(quantile_levels) array of predicted conditional quantiles
    """
    cause = np.asarray(cause)
    if cause.ndim == 1:
        cause = cause.reshape(-1, 1)
    model.fit(cause, effect)
    pred = model.predict(
        cause, output_type="full", quantiles=list(quantile_levels)
    )
    return pred["mean"], np.column_stack(pred["quantiles"])


def _quantile_residual(
    model,
    cause,
    effect,
    q_lo,
    q_hi,
    z_lo,
    z_hi,
    qr_mean=False,
    scale_method="quantile_range",
    gl_m=10,
):
    """Location-scale residual of effect | cause, from TabPFN quantiles.

    Parameters
    ----------
    - model: a `tabpfn.TabPFNRegressor` instance
    - cause: predictor variable(s) ((n,) or (n, p))
    - effect: target variable (n,)
    - q_lo, q_hi: lower/upper quantile levels used for the scale estimate when
      `scale_method="quantile_range"`; ignored otherwise
    - z_lo, z_hi: quantiles used for normalization ("quantile_range" variant)
    - qr_mean: False (default), uses median quantile to estimate the location when
      `scale_method="quantile_range"`; ignored otherwise
    - scale_method: "quantile_range" (default, uses `quantile_range`) or "gauss_legendre"
    - gl_m: number of Gauss-Legendre quadrature nodes when `scale_method="gauss_legendre"`
      (default 100); ignored otherwise

    Returns
    -------
    - resid: (effect - loc) / scale (1D NumPy array)
    - loc: predicted median of effect | cause
    - scale: scale estimate per `scale_method` (1D NumPy array)
    """
    if scale_method == "quantile_range":
        if qr_mean:
            loc, preds = _fit_conditional_quantiles_and_mean(
                model, cause, effect, [q_lo, 0.5, q_hi]
            )
        else:
            preds = _fit_conditional_quantiles(
                model, cause, effect, [q_lo, 0.5, q_hi]
            )
            loc = preds[:, 1]
        scale = np.maximum((preds[:, 2] - preds[:, 0]) / (z_hi - z_lo), 1e-8)
    elif scale_method == "gauss_legendre":
        return _quantile_residual_integrated(model, cause, effect, m=gl_m)
    else:
        raise ValueError(f"Unknown scale_method: {scale_method!r}")

    resid = (effect - loc) / scale
    return resid, loc, scale


def _quantile_residual_integrated(model, cause, effect, m=10):
    """Location-scale residual of effect | cause, with the scale via Gauss-Legendre.

    Parameters
    ----------
    - model: a `tabpfn.TabPFNRegressor` instance
    - cause: predictor variable(s) ((n,) or (n, p))
    - effect: target variable (n,)
    - m: number of Gauss-Legendre quadrature nodes (integer, default=10)

    Returns
    -------
    - resid: (effect - loc) / scale (1D NumPy array)
    - loc: predicted mean of effect | cause (1D NumPy array)
    - scale: sqrt of the quantile-integrated variance estimate (1D NumPy array)
    """
    nodes, weights = _quadrature_nodes(m)
    loc, quant_by_point = _fit_conditional_quantiles_and_mean(
        model, cause, effect, nodes
    )
    variance = np.sum(weights * (quant_by_point - loc[:, None]) ** 2, axis=1)
    scale = np.maximum(np.sqrt(variance), 1e-8)
    resid = (effect - loc) / scale
    return resid, loc, scale


def Loscar_bivariate(
    x,
    y,
    scale_quantiles=(0.05, 0.95),
    tabpfn_kwargs=None,
    random_state=0,
    model=None,
    qr_mean=False,
    scale_method="quantile_range",
    gl_m=10,
    device="cpu",
):
    """Test causal direction via TabPFN quantile residuals + HSIC.

    Parameters
    ----------
    - x: covariate (n x 1 or (n,) NumPy array)
    - y: outcome (n x 1 or (n,) NumPy array)
    - scale_quantiles: (low, high) lower/upper quantile levels used for the
      scale estimate when `scale_method="quantile_range"`; ignored otherwise
    - tabpfn_kwargs: keyword arguments for `tabpfn.TabPFNRegressor`,
      ignored if `model` is given (dict, default=None); see
      `make_tabpfn_regressor`
    - random_state: seed for TabPFN's default random state, ignored if
      `model` is given (integer, default=0)
    - model: an existing `tabpfn.TabPFNRegressor`
    - scale_method: "quantile_range" (default, uses `quantile_range`) or "gauss_legendre"
    - gl_m: number of Gauss-Legendre quadrature nodes when `scale_method="gauss_legendre"`
      (default 10); ignored otherwise

    Returns
    -------
    - indep_score: HSIC(y, resid_RV) - HSIC(x, resid_FW); positive favors x -> y
    - indep_dir: 1 if x -> y is favored, 0 if y -> x is favored
    - res_FW: dict with the x -> y hypothesis's cause/effect, predicted
      location/scale, residual, and HSIC statistic
    - res_RV: dict with the y -> x hypothesis's analogous outputs
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if model is None:
        model = make_tabpfn_regressor(tabpfn_kwargs, random_state, device)

    q_lo, q_hi = scale_quantiles
    z_lo, z_hi = norm.ppf(q_lo), norm.ppf(q_hi)

    resid_fw, loc_fw, scale_fw = _quantile_residual(
        model,
        x,
        y,
        q_lo,
        q_hi,
        z_lo,
        z_hi,
        qr_mean,
        scale_method=scale_method,
        gl_m=gl_m,
    )
    resid_rv, loc_rv, scale_rv = _quantile_residual(
        model,
        y,
        x,
        q_lo,
        q_hi,
        z_lo,
        z_hi,
        qr_mean,
        scale_method=scale_method,
        gl_m=gl_m,
    )

    indep_fw = hsic_test_gamma(x, resid_fw)[0]
    indep_rv = hsic_test_gamma(y, resid_rv)[0]

    indep_score = indep_rv - indep_fw
    indep_dir = 1 if indep_score > 0 else 0

    res_FW = {
        "cause": x,
        "effect": y,
        "loc": loc_fw,
        "scale": scale_fw,
        "resid": resid_fw,
        "indep": indep_fw,
    }
    res_RV = {
        "cause": y,
        "effect": x,
        "loc": loc_rv,
        "scale": scale_rv,
        "resid": resid_rv,
        "indep": indep_rv,
    }
    return indep_score, indep_dir, res_FW, res_RV
