"""Run one Loscar or RESIT variant (--method) across a grid of (d, sem_type, noise_dist)
LSNM/heteroscedastic SEM settings.

The baselines are run on the same grid by benchmark_multivariate_synthetic_baselines.py.
"""
# Don't set CUDA_VISIBLE_DEVICES="": cdt (used for the metrics) parses it with ast.literal_eval
# at import time and fails on an empty string. Use --device instead.

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# Repo root (for `data`) and experiments/ (for `metrics`); loscar itself is pip-installed.
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXPERIMENTS_DIR.parent
for _path in (_REPO_ROOT, _EXPERIMENTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from data.dgp import simulate_dag, simulate_nonlinear_sem, set_random_seed  # noqa: E402
from loscar.loscar import Loscar  # noqa: E402
from loscar.resit import RESIT  # noqa: E402
from loscar.mean_regressors import TabPFNMeanRegressor, SplineRegression  # noqa: E402
from loscar.loscar_bivariate import make_tabpfn_regressor  # noqa: E402
from metrics import compute_metrics  # noqa: E402

DEFAULT_OUT_DIR = "experiments/results"
CHECKPOINT_EVERY = 10

D_VALUES = [6, 12]
SEM_TYPES = ["AN-s", "LS-s"]
# user-facing name -> internal noise_dist string; "mix" == "coinflip" (per-node random
# gauss/skew-t/exponential)
NOISE_DISTS = {"gauss": "gauss", "exponential": "exponential", "skew-t": "skew-t", "mix": "coinflip"}
N_SAMPLES = 1000
HET_NOISE_FRACTION = 1.0  # no-op for AN/AN-s; makes every non-source node heteroscedastic under LS/LS-s
GLOBAL_SEED = 0

METHODS = ["Loscar", "Loscar-GL", "Loscar-FGLS", "RESIT-TabPFN", "RESIT-spline"]
# methods that share the pretrained TabPFN regressor
TABPFN_METHODS = ("Loscar", "Loscar-GL", "RESIT-TabPFN")

MODEL_VERSIONS = {
    "v3": "tabpfn-v3-regressor-v3_default.ckpt",
    "v35": "tabpfn-v3.5-20260909.safetensors",
    "v35-fast": "tabpfn-v3.5-fast-20260909.safetensors",
}
# TabPFN kwargs, pinned to the v3 regressor checkpoint
TABPFN_KWARGS = dict(random_state=0, inference_precision=torch.float32, model_path=MODEL_VERSIONS["v3"])

# pretrained model, loaded once per process and reused across all items
_TABPFN_MODEL = None
_TABPFN_DEVICE = None


def _init_tabpfn_model(device):
    global _TABPFN_MODEL, _TABPFN_DEVICE
    _TABPFN_DEVICE = device
    _TABPFN_MODEL = make_tabpfn_regressor({**TABPFN_KWARGS, "device": _TABPFN_DEVICE})


# ---------------------------------------------------------------------------
# Method dispatch: each returns (dag_or_None, cpdag_or_None, pi_est_or_None),
# with row=cause/col=effect adjacency.
# ---------------------------------------------------------------------------


def _run_loscar_gl(X):
    model = Loscar(
        model=_TABPFN_MODEL,
        device=_TABPFN_DEVICE,
        scale_method="gauss_legendre",
        gl_m=10,
    )
    model.fit(X)
    dag = (model.adjacency_matrix_ != 0).astype(int)
    return dag, None, list(model.causal_order_)


def _run_loscar(X):
    model = Loscar(
        model=_TABPFN_MODEL,
        device=_TABPFN_DEVICE,
        scale_method="quantile_range",
        scale_quantiles=(0.05, 0.95),
    )
    model.fit(X)
    dag = (model.adjacency_matrix_ != 0).astype(int)
    return dag, None, list(model.causal_order_)


def _run_loscar_fgls(X):
    # spline-basis IFGLS location-scale regression (CPU only)
    model = Loscar(regressor="fgls")
    model.fit(X)
    dag = (model.adjacency_matrix_ != 0).astype(int)
    return dag, None, list(model.causal_order_)


def _run_resit_spline(X):
    # original additive-noise RESIT (no location-scale standardization) with a spline mean
    # regressor
    model = RESIT(regressor=SplineRegression())
    model.fit(X)
    dag = (model.adjacency_matrix_ != 0).astype(int)
    return dag, None, list(model.causal_order_)


def _run_resit_tabpfn(X):
    # original additive-noise RESIT with TabPFN's predicted mean as regressor
    model = RESIT(regressor=TabPFNMeanRegressor(_TABPFN_MODEL))
    model.fit(X)
    dag = (model.adjacency_matrix_ != 0).astype(int)
    return dag, None, list(model.causal_order_)


_METHOD_FNS = {"Loscar": _run_loscar, "Loscar-GL": _run_loscar_gl, "Loscar-FGLS": _run_loscar_fgls,
               "RESIT-TabPFN": _run_resit_tabpfn, "RESIT-spline": _run_resit_spline}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def simulate_dataset(d, sem_type, noise_dist, seed):
    """Ground-truth DAG and standardized LSNM sample matrix for one replicate (deterministic in `seed`)."""
    set_random_seed(seed)
    B_true = simulate_dag(d=d, s0=d, graph_type="ER")
    X, _, _, _ = simulate_nonlinear_sem(
        N_SAMPLES, sem_type=sem_type, noise_dist=noise_dist, het_noise_fraction=HET_NOISE_FRACTION, B=B_true
    )
    X = StandardScaler().fit_transform(X)
    return X, B_true


# ---------------------------------------------------------------------------
# Worker: one (d, sem_type, noise_dist, replicate, method) fit = one unit of work
# ---------------------------------------------------------------------------
def _worker(item):
    d, sem_type, noise_dist_name, rep, data_seed, method_name = item
    noise_dist = NOISE_DISTS[noise_dist_name]

    X, B_true = simulate_dataset(d, sem_type, noise_dist, data_seed)

    row = {
        "d": d,
        "s0": d,
        "sem_type": sem_type,
        "noise_dist": noise_dist_name,
        "rep": rep,
        "method": method_name,
    }
    t0 = time.time()
    try:
        # re-seed so methods with internal randomness are reproducible per replicate
        set_random_seed(data_seed)
        dag, cpdag, pi_est = _METHOD_FNS[method_name](X)
        row["runtime_sec"] = time.time() - t0
        row["error"] = ""
        row.update(compute_metrics(B_true, dag, cpdag, pi_est))
    except Exception as e:  # noqa: BLE001  (a huge grid job shouldn't die on one cell)
        row["runtime_sec"] = time.time() - t0
        row["error"] = f"{type(e).__name__}: {e}"
        row["_traceback"] = traceback.format_exc()

    return row


def build_work_items(method_name, n_datasets, rep_start, rep_end, d_values=None, sem_types=None, noise_dist_names=None):
    d_values = D_VALUES if d_values is None else d_values
    sem_types = SEM_TYPES if sem_types is None else sem_types
    noise_dist_names = list(NOISE_DISTS) if noise_dist_names is None else noise_dist_names
    items = []
    combo_idx = 0
    for d in d_values:
        for sem_type in sem_types:
            for noise_dist_name in noise_dist_names:
                for rep in range(rep_start, rep_end):
                    data_seed = GLOBAL_SEED + combo_idx * n_datasets + rep
                    items.append((d, sem_type, noise_dist_name, rep, data_seed, method_name))
                combo_idx += 1
    return items


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=METHODS,
        help="Causal-discovery method to run this invocation. Run the script once per method.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the TabPFN-based methods, e.g. 'cpu', 'cuda', 'cuda:0' (default: cpu). "
        "Ignored by Loscar-FGLS and RESIT-spline.",
    )
    parser.add_argument(
        "--n-datasets",
        type=int,
        default=100,
        help="Replicate datasets per (d, sem_type, noise_dist) combo (default: 100).",
    )
    parser.add_argument(
        "--rep-start", type=int, default=0, help="First replicate index to run (for sharding, default: 0)."
    )
    parser.add_argument(
        "--rep-end",
        type=int,
        default=None,
        help="One past the last replicate index to run (default: --n-datasets).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run just one (d, sem_type, noise_dist, replicate) to check for errors, then exit. "
        "Writes to a separate *_smoketest.csv so it can't collide with a real run's output.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Directory to write result CSVs to (default: {DEFAULT_OUT_DIR}). Use this to keep "
        "one method/run's shards in their own subfolder rather than mixed in with others.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rep_end = args.rep_end if args.rep_end is not None else args.n_datasets

    out_dir = Path(args.out_dir) if args.out_dir else Path(DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)

    set_random_seed(GLOBAL_SEED)

    if args.method in TABPFN_METHODS:
        _init_tabpfn_model(args.device)

    if args.smoke_test:
        out_path = out_dir / f"classical_grid_experiment_{args.method}_smoketest.csv"
        work_items = build_work_items(
            args.method,
            1,
            0,
            1,
            d_values=D_VALUES[:1],
            sem_types=SEM_TYPES[:1],
            noise_dist_names=list(NOISE_DISTS)[:1],
        )
        print(f"SMOKE TEST [{args.method}]: {len(work_items)} work item(s)", flush=True)
    else:
        # shards get their own file so concurrent processes don't overwrite each other
        is_sharded = args.rep_start != 0 or rep_end != args.n_datasets
        shard_suffix = f"_rep{args.rep_start}-{rep_end}" if is_sharded else ""
        out_path = out_dir / f"classical_grid_experiment_{args.method}{shard_suffix}.csv"
        work_items = build_work_items(args.method, args.n_datasets, args.rep_start, rep_end)
        print(f"Total work items: {len(work_items)} [{args.method}]", flush=True)

    rows = []
    t_start = time.time()
    for i, item in enumerate(work_items, 1):
        row = _worker(item)
        if row["error"]:
            print(f"ERROR [{i}/{len(work_items)}] {row}", flush=True)
        rows.append(row)
        if i % CHECKPOINT_EVERY == 0 or i == len(work_items):
            pd.DataFrame(rows).to_csv(out_path, index=False)
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(work_items) - i) / rate if rate > 0 else float("nan")
            print(
                f"[{i}/{len(work_items)}] elapsed={elapsed / 60:.1f}min rate={rate:.2f}/s eta={eta / 60:.1f}min",
                flush=True,
            )

    print(f"\nsaved: {out_path}")
    if args.smoke_test:
        errors = [r for r in rows if r["error"]]
        print(f"\nSMOKE TEST {'FAILED' if errors else 'PASSED'}: {len(errors)}/{len(rows)} item(s) errored.")
