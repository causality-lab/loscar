"""Graph metrics shared by the multivariate and real-world benchmarks."""

import os
import platform
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import cdt
import cdt.metrics as cdt_metrics

# On Windows, point cdt at R_HOME's Rscript.exe if set (cdt otherwise looks up "Rscript" on PATH).
if platform.system() == "Windows":
    _r_home = os.environ.get("R_HOME")
    if _r_home:
        _rscript_path = Path(_r_home) / "bin" / "Rscript.exe"
        if _rscript_path.is_file():
            cdt.SETTINGS.rpath = str(_rscript_path)

METRIC_KEYS = [
    "precision",
    "recall",
    "F1",
    "order_errors",
    "SHD_DAG",
    "SID_DAG",
    "SHD_CPDAG",
    "SID_CPDAG_lower",
    "SID_CPDAG_upper",
]


def _binarize_adj(A):
    A = (np.asarray(A) > 0).astype(int)
    np.fill_diagonal(A, 0)
    return A


def num_errors(order, adj):
    """Number of true edges that go backward relative to `order`: for each
    position i, count true edges from any later-ordered node into
    order[i].
    """
    A = np.asarray(adj)
    order = np.asarray(order, dtype=int)
    err = 0
    for i in range(len(order)):
        err += A[order[i + 1 :], order[i]].sum()
    return int(err)


def compute_metrics(true_adj, dag, cpdag, pi_est):
    """Precision/recall/F1 (directed-edge) plus order_errors, evaluated
    against `dag` when the method commits to a full orientation, else
    against `cpdag`. `order_errors` is NaN when `pi_est` is None.

    Also reports `cdt.metrics`' R-backed SHD/SID (on `dag`) and
    SHD_CPDAG/SID_CPDAG (on `cpdag`); these are NaN if R or the
    `pcalg`/`SID` packages are unavailable.
    """
    est_adj = dag if dag is not None else cpdag
    A_true = _binarize_adj(true_adj)
    A_hat = _binarize_adj(est_adj)

    y_true = A_true.ravel()
    y_pred = A_hat.ravel()
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "order_errors": (
            num_errors(pi_est, A_true) if pi_est is not None else np.nan
        ),
    }

    # SHD is pure Python; SID needs R. Separate try/except blocks, so an R
    # failure for one group doesn't discard the other.
    if dag is not None:
        metrics["SHD_DAG"] = float(cdt_metrics.SHD(true_adj, dag))
    try:
        if dag is not None:
            metrics["SID_DAG"] = float(cdt_metrics.SID(true_adj, dag))
    except Exception as exc:
        print(f"  cdt.metrics (R-backed, DAG) FAILED, reporting NaN: {exc}")
        metrics["SID_DAG"] = np.nan
    if cpdag is not None:
        metrics["SHD_DAG"] = float(cdt_metrics.SHD(true_adj, cpdag))
    try:
        if cpdag is not None:
            metrics["SID_DAG"] = float(cdt_metrics.SID(true_adj, cpdag))
            metrics["SHD_CPDAG"] = float(
                cdt_metrics.SHD_CPDAG(true_adj, cpdag)
            )
            lower, upper = cdt_metrics.SID_CPDAG(true_adj, cpdag)
            metrics["SID_CPDAG_lower"] = float(lower)
            metrics["SID_CPDAG_upper"] = float(upper)
        else:
            metrics["SHD_CPDAG"] = np.nan
            metrics["SID_CPDAG_lower"] = np.nan
            metrics["SID_CPDAG_upper"] = np.nan
    except Exception as exc:
        print(f"  cdt.metrics (R-backed, CPDAG) FAILED, reporting NaN: {exc}")
        for key in ["SHD_CPDAG", "SID_CPDAG_lower", "SID_CPDAG_upper"]:
            metrics[key] = np.nan

    return metrics
