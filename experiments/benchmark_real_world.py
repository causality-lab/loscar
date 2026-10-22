"""Run the Loscar variants on the real-world datasets (SACHS, LightTunnel).

The baselines are run by benchmark_real_world_baselines.py; see
data/real_world.py for the datasets, their references and licenses.
"""

import os
import sys
import time
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import torch

from loscar.loscar import Loscar
from loscar.loscar_bivariate import make_tabpfn_regressor

# Repo root (for data/); loscar itself is pip-installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.real_world import DATASETS  # noqa: E402
from metrics import METRIC_KEYS, compute_metrics  # noqa: E402

MODEL_VERSIONS = {
    "v3": "tabpfn-v3-regressor-v3_default.ckpt",
    "v35": "tabpfn-v3.5-20260909.safetensors",
    "v35-fast": "tabpfn-v3.5-fast-20260909.safetensors",
}


def get_device():
    """Check available devices and return preferred one: CUDA > CPU."""
    cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return cuda_devices[0] if cuda_devices else "cpu"


def method_loscar_gl(X, device, tabpfn_kwargs):
    tabpfn_model = make_tabpfn_regressor({**tabpfn_kwargs, "device": str(device)})
    model = Loscar(
        model=tabpfn_model, scale_method="gauss_legendre", device=device
    )
    model.fit(X)
    dag = (model.adjacency_matrix_ != 0).astype(int)
    return dag, None, list(model.causal_order_)


# Each method takes (X, device, configs), where configs holds one config per method.
METHODS = {
    "Loscar-GL": lambda X, device, configs: method_loscar_gl(
        X, device, configs["Loscar-GL"]
    ),
}


# --------------------------------------------------------------------------
# Per-dataset run + storage (one row per method).
# --------------------------------------------------------------------------


def run_dataset(ds_name, loader, device, configs, out_dir):
    X, true_adj, _ = loader()
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # standardize
    meta = {
        "n": X.shape[0],
        "d": X.shape[1],
        "true_edges": int(true_adj.sum()),
    }
    print(
        f"=== {ds_name}: n={meta['n']}, d={meta['d']}, "
        f"true edges={meta['true_edges']} ==="
    )

    rows = []
    for method_name, fn in METHODS.items():
        t0 = time.time()
        try:
            dag, cpdag, pi_est = fn(X, device, configs)
            metrics = compute_metrics(true_adj, dag, cpdag, pi_est)
            save_prediction(
                out_dir, ds_name, method_name, true_adj, dag, cpdag, pi_est
            )
        except Exception as exc:
            print(f"  {method_name}: FAILED ({exc})")
            metrics = {k: np.nan for k in METRIC_KEYS}
        metrics["runtime_s"] = time.time() - t0
        print(
            f"  {method_name} ({metrics['runtime_s']:.1f}s): "
            f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"F1={metrics['F1']:.3f} "
            f"order_errors={metrics['order_errors']} | "
            f"SHD_DAG={metrics['SHD_DAG']} SID_DAG={metrics['SID_DAG']:.1f} "
            f"SHD_CPDAG={metrics['SHD_CPDAG']} "
            f"SID_CPDAG=[{metrics['SID_CPDAG_lower']:.1f}, "
            f"{metrics['SID_CPDAG_upper']:.1f}]"
        )
        rows.append(
            {"Method": method_name, "Dataset": ds_name, **meta, **metrics}
        )
    return pd.DataFrame(rows)


def save_prediction(
    out_dir, ds_name, method_name, true_adj, dag, cpdag, pi_est
):
    """Save a method's predicted graph(s) so additional metrics can be
    computed later without rerunning causal discovery."""
    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    path = os.path.join(pred_dir, f"{ds_name}_{method_name}.npz")
    arrays = {"true_adj": true_adj, "cpdag": cpdag}
    if dag is not None:
        arrays["dag"] = dag
    if pi_est is not None:
        arrays["pi_est"] = np.asarray(pi_est)
    np.savez(path, **arrays)
    return path


def dataset_result_path(out_dir, ds_name):
    return os.path.join(out_dir, f"real_world_loscar_{ds_name}.csv")


def save_dataset_results(out_dir, ds_name, df):
    path = dataset_result_path(out_dir, ds_name)
    df.to_csv(path, index=False)
    print(f"Saved {ds_name} results to {path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=sorted(DATASETS) + ["all"],
        help=(
            "Which dataset to run (default: all). Each dataset's results "
            "are stored in their own file, so e.g. SACHS and LightTunnel "
            "can be run independently (separate invocations/processes) "
            "and merged afterwards."
        ),
    )
    parser.add_argument(
        "--model-version", default="v3", choices=sorted(MODEL_VERSIONS)
    )
    parser.add_argument(
        "--device",
        default=None,
        help="e.g. cuda:0, cuda:1, cpu (default: auto-detect)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--results-directory",
        type=str,
        default="experiments/results/real_world",
        help=(
            "Folder in which the results should be stored, one "
            "'real_world_loscar_<dataset>.csv' per dataset plus a "
            "'predictions/' subfolder."
        ),
    )
    parser.add_argument(
        "--ignore-pretraining-limits",
        action="store_true",
        help=(
            "TabPFN refuses to run on CPU with more than 1000 samples by "
            "default (a speed guard, not a capacity limit - it does not "
            "apply on CUDA). Pass this to override it, e.g. for "
            "LightTunnel (n=10000) on CPU."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device or get_device())
    tabpfn_kwargs = {
        "model_path": MODEL_VERSIONS[args.model_version],
        "random_state": args.random_state,
        "inference_precision": torch.float32,
        "ignore_pretraining_limits": args.ignore_pretraining_limits,
    }
    configs = {"Loscar-GL": tabpfn_kwargs}
    out_dir = args.results_directory
    os.makedirs(out_dir, exist_ok=True)

    df = None
    selected = DATASETS if args.dataset == "all" else [args.dataset]
    for ds_name in selected:
        file_name = "all" if args.dataset == "all" else ds_name
        df_new = run_dataset(
            ds_name, DATASETS[ds_name], device, configs, out_dir
        )
        if df is None:
            df = df_new
        else:
            df = pd.concat([df, df_new], ignore_index=True)
        save_dataset_results(out_dir, file_name, df)


if __name__ == "__main__":
    main()
