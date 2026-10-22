import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import SplineTransformer

# expected value of log chi square with 1 df
# Harvey correction within FGLS: digamma(0.5) - log(0.5) up to 15 digit
EV_LOG_CHI2 = -1.27036284546148

def ifgls(y, Phi, Psi, n_steps=100, log_eps=1e-6, ridge_alpha_mean=1e-6, ridge_alpha_var=1e-6):
    """Carry out iterated FGLS with feature maps Phi, Psi for mean and variance, respectively.

    Model: y = Phi @ beta + exp(0.5*Psi @ gamma) * e

    Returns: hat_beta, hat_gamma
    """
    hat_beta_list = list()
    hat_gamma_list = list()
    # Mean fit
    mean_ols = sm.OLS(y, Phi).fit_regularized(L1_wt=0.0, alpha=ridge_alpha_mean)
    hat_beta = mean_ols.params
    hat_beta_list.append(hat_beta)
    hat_f = Phi @ hat_beta
    hat_u_sq = np.square(y - hat_f)
    hat_log_u_sq = np.log(hat_u_sq + log_eps)
    # Variance fit
    sd_ols = sm.OLS(hat_log_u_sq, Psi).fit_regularized(L1_wt=0.0, alpha=ridge_alpha_var)
    hat_gamma = sd_ols.params
    hat_gamma_list.append(hat_gamma)
    hat_g_sq = np.clip(np.exp(Psi @ hat_gamma), a_min=1e-7, a_max=1e7)
    hat_g = np.sqrt(hat_g_sq)
    # FGLS fit
    fgls_mean = sm.GLS(y, Phi, sigma=hat_g).fit_regularized(L1_wt=0.0, alpha=ridge_alpha_mean)
    hat_beta = fgls_mean.params
    hat_f = Phi @ hat_beta

    # Iterate:
    if n_steps > 1:
        for _ in range(n_steps - 1):
            # Prepare variance refit
            hat_u_sq = np.square(y - hat_f)
            hat_log_u_sq = np.log(hat_u_sq)
            # Variance refit
            sd_ols = sm.OLS(hat_log_u_sq, Psi).fit_regularized(L1_wt=0.0, alpha=ridge_alpha_var)
            hat_gamma = sd_ols.params
            hat_gamma_list.append(hat_gamma)
            hat_g_sq = np.clip(np.exp(Psi @ hat_gamma), a_min=1e-7, a_max=1e7)
            hat_g = np.sqrt(hat_g_sq)
            # FGLS refit
            fgls_mean = sm.GLS(y, Phi, sigma=hat_g).fit_regularized(L1_wt=0.0, alpha=ridge_alpha_mean)
            hat_beta = fgls_mean.params
            hat_beta_list.append(hat_beta)
            hat_f = Phi @ hat_beta
            # Check convergence by change of hat_beta
            delta_beta = max(abs(hat_beta_list[-1] - hat_beta_list[-2]))
            delta_gamma = max(abs(hat_gamma_list[-1] - hat_gamma_list[-2]))
            if max(delta_beta, delta_gamma) < 1e-5:
                break

    return hat_beta, hat_gamma


def het_fit_ifgls(y, X, n_steps=100, log_eps=1e-6,
                  n_knots_mean=6, n_knots_var=4,
                  ridge_alpha_mean=1e-4, ridge_alpha_var=1e-2):
    """Fit heteroscedastic model using IFGLS."""
    feature_map_mean = SplineTransformer(n_knots=n_knots_mean, degree=3,
                                         include_bias=False, extrapolation="linear")
    feature_map_var = SplineTransformer(n_knots=n_knots_var, degree=3,
                                        include_bias=False, extrapolation="linear")
    Phi_X = feature_map_mean.fit_transform(X)
    Psi_X = feature_map_var.fit_transform(X)
    Phi = np.column_stack([np.ones(Phi_X.shape[0]), Phi_X])
    Psi = np.column_stack([np.ones(Psi_X.shape[0]), Psi_X])
    hat_beta, hat_gamma = ifgls(y, Phi, Psi, n_steps, log_eps, ridge_alpha_mean, ridge_alpha_var)

    hat_gamma = hat_gamma.copy()
    hat_gamma[0] -= EV_LOG_CHI2  # debias the variance estimate

    def f(X_):
        """Predict mean and standard deviation."""
        Phi_X = feature_map_mean.transform(X_)
        Psi_X = feature_map_var.transform(X_)
        Phi = np.column_stack([np.ones(Phi_X.shape[0]), Phi_X])
        Psi = np.column_stack([np.ones(Psi_X.shape[0]), Psi_X])
        hat_g_sq = np.clip(np.exp(Psi @ hat_gamma), a_min=1e-7, a_max=1e7)
        scale = np.sqrt(hat_g_sq)
        loc = Phi @ hat_beta
        return loc, scale

    return f


def _fgls_residual(cause, effect, fgls_kwargs):
    """Location-scale residual of effect | cause, from `ifgls_spline.het_fit_ifgls` (a spline-
    basis IFGLS fit) instead of TabPFN quantile regression.

    Parameters
    ----------
    - cause: predictor variable(s), (n, p)
    - effect: target variable (n,)
    - fgls_kwargs: dict of extra keyword arguments forwarded to `het_fit_ifgls` (n_steps,
      n_knots_mean, n_knots_var, ridge_alpha_mean, ridge_alpha_var, log_eps)

    Returns
    -------
    - resid: (effect - loc) / scale (1D NumPy array)
    - loc: predicted mean of effect | cause (1D NumPy array)
    - scale: predicted std of effect | cause (1D NumPy array)
    """
    kwargs = {"n_steps": 1, **fgls_kwargs}
    predict = het_fit_ifgls(effect, cause, **kwargs)
    loc, scale = predict(cause)
    scale = np.maximum(scale, 1e-8)
    resid = (effect - loc) / scale
    return resid, loc, scale