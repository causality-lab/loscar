"""Run a bivariate causal-direction method over one benchmark set of pairs.

Methods (pass one via --method, default "Loscar"):

- Loscar: Loscar_bivariate, 5%-95% quantile-range scale estimate
- Loscar-GL: Loscar_bivariate, Gauss-Legendre scale estimate (m=10)
- Loscar-FGLS: location-scale residuals from spline IFGLS
- RESIT-TabPFN: additive-noise residuals from TabPFN's predicted mean
- RESIT-spline: additive-noise residuals from a spline regression

Benchmark sets (pass one via --benchmark):

- anlsmn: the ANLSMN benchmark - AN, ANs, LS, LSs, MNU (100 pairs each)
- tuebingen: the Tuebingen benchmark (108 pairs)
- sim: the Benchmark_simulated suite - SIM, SIMc, SIMG, SIMln
  (100 pairs each)
- dataverse: the Dataverse pairs - Cha, Multi, Net (300 pairs each)
- skew: the skewANsLSs grid - {ANs, LSs} x {AGN, SN} distribution x
  skewness level (100 pairs per combination)

Assumes datasets to be downloaded and stored in data/ (see data/datasets.py).
"""
import sys
import argparse
import csv
from pathlib import Path
import pickle
import time
import numpy as np

import torch
from sklearn.preprocessing import StandardScaler
from loscar.loscar_bivariate import make_tabpfn_regressor  # noqa: E402
from loscar.loscar_bivariate import Loscar_bivariate
from loscar.hsic import hsic_test_gamma  # noqa: E402
from loscar.mean_regressors import SplineRegression, TabPFNMeanRegressor  # noqa: E402
from loscar.ifgls_spline import _fgls_residual  # noqa: E402

# Repo root (for data/); loscar itself is pip-installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.datasets import (  # noqa: E402
    AN,
    ANs,
    Cha,
    LS,
    LSs,
    MNU,
    Multi,
    Net,
    SIM,
    SIMc,
    SIMG,
    SIMln,
    Tuebingen,
    skewANsLSs,
)

GROUND_TRUTH = 1  # dataset.cause/.effect are always already in the true order


def _run_loscar(x, y, random_state, model):
    score, direction, res_FW, res_RV = Loscar_bivariate(
        x, y, random_state=random_state, model=model
    )
    info = {
        "indep_fw": res_FW["indep"],
        "indep_rv": res_RV["indep"],
    }
    return score, direction, info


def _run_loscar_gl(x, y, random_state, model):
    score, direction, res_FW, res_RV = Loscar_bivariate(
        x,
        y,
        random_state=random_state,
        model=model,
        scale_method="gauss_legendre",
        gl_m=10,
    )
    info = {
        "indep_fw": res_FW["indep"],
        "indep_rv": res_RV["indep"],
    }
    return score, direction, info


def _an_direction(x, y, reg_fw, reg_rv):
    """Shared decision rule for the plain-residual (AN) methods below."""
    x, y = (
        np.asarray(x).flatten(),
        np.asarray(y).flatten(),
    )
    reg_fw.fit(x, y)
    indep_fw = hsic_test_gamma(x, y - reg_fw.predict(x))[0]

    reg_rv.fit(y, x)
    indep_rv = hsic_test_gamma(y, x - reg_rv.predict(y))[0]

    score = (
        indep_rv - indep_fw
    )  # positive favors x -> y (fw residual more independent)
    direction = 1 if score > 0 else 0
    return score, direction, indep_fw, indep_rv


def _run_resit_spline(x, y, random_state, model):
    score, direction, indep_fw, indep_rv = _an_direction(
        x, y, SplineRegression(), SplineRegression()
    )
    return score, direction, {"indep_fw": indep_fw, "indep_rv": indep_rv}


def _run_resit_tabpfn(x, y, random_state, model):
    if model is None:
        model = make_tabpfn_regressor(random_state=random_state)
    reg = TabPFNMeanRegressor(model)
    score, direction, indep_fw, indep_rv = _an_direction(x, y, reg, reg)
    return score, direction, {"indep_fw": indep_fw, "indep_rv": indep_rv}


def _run_loscar_fgls(x, y, random_state, model, fgls_kwargs=None):
    fgls_kwargs = fgls_kwargs or {}
    x_flat, y_flat = (
        np.asarray(x).flatten(),
        np.asarray(y).flatten(),
    )  # het_fit_ifgls needs a 1D target

    resid_fw, *_ = _fgls_residual(x_flat.reshape(-1, 1), y_flat, fgls_kwargs)
    indep_fw = hsic_test_gamma(x_flat, resid_fw)[0]

    resid_rv, *_ = _fgls_residual(y_flat.reshape(-1, 1), x_flat, fgls_kwargs)
    indep_rv = hsic_test_gamma(y_flat, resid_rv)[0]

    score = indep_rv - indep_fw
    direction = 1 if score > 0 else 0
    return score, direction, {"indep_fw": indep_fw, "indep_rv": indep_rv}


METHODS = {
    "Loscar": _run_loscar,
    "Loscar-GL": _run_loscar_gl,
    "Loscar-FGLS": _run_loscar_fgls,
    "RESIT-TabPFN": _run_resit_tabpfn,
    "RESIT-spline": _run_resit_spline,
}


def get_device():
    """Check available devices and return preferred one: CUDA > CPU."""
    cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return cuda_devices[0] if cuda_devices else "cpu"


def _anlsmn_tasks():
    classes = {"AN": AN, "ANs": ANs, "LS": LS, "LSs": LSs, "MNU": MNU}
    for group, cls in classes.items():
        for pid in range(1, cls.n_datasets + 1):
            yield group, pid, lambda cls=cls, pid=pid: cls(
                pid, preprocessor=StandardScaler(), double=True
            )


def _tuebingen_tasks():
    for pid in range(1, Tuebingen.n_datasets + 1):
        yield "Tuebingen", pid, lambda pid=pid: Tuebingen(
            pid, preprocessor=StandardScaler(), double=True
        )


def _sim_tasks():
    classes = {"SIM": SIM, "SIMc": SIMc, "SIMG": SIMG, "SIMln": SIMln}
    for group, cls in classes.items():
        for pid in range(1, cls.n_datasets + 1):
            yield group, pid, lambda cls=cls, pid=pid: cls(
                pid, preprocessor=StandardScaler(), double=True
            )


def _dataverse_tasks():
    classes = {"Cha": Cha, "Multi": Multi, "Net": Net}
    for group, cls in classes.items():
        for pid in range(1, cls.n_datasets + 1):
            yield group, pid, lambda cls=cls, pid=pid: cls(
                pid, preprocessor=StandardScaler(), double=True
            )


def _skew_tasks():
    skewness_levels = {
        "AGN": ["med", "strong", "extreme"],
        "SN": ["med", "strong"],
    }
    for dataset_type in ["ANs", "LSs"]:
        for distribution in ["AGN", "SN"]:
            for level in skewness_levels[distribution]:
                group_parts = []
                group_parts += [dataset_type, distribution, level]
                group = "_".join(group_parts)
                for pid in range(1, skewANsLSs.n_datasets + 1):
                    yield (
                        group,
                        pid,
                        lambda dt=dataset_type, dist=distribution, lvl=level, pid=pid: skewANsLSs(
                            dataset_type=dt,
                            distribution=dist,
                            skewness_level=lvl,
                            pair_id=pid,
                            preprocessor=StandardScaler(),
                            double=True,
                        ),
                    )


BENCHMARK_SETS = {
    "anlsmn": _anlsmn_tasks,
    "tuebingen": _tuebingen_tasks,
    "sim": _sim_tasks,
    "dataverse": _dataverse_tasks,
    "skew": _skew_tasks,
}

MODEL_VERSIONS = {
    "v3": "tabpfn-v3-regressor-v3_default.ckpt",
    "v35": "tabpfn-v3.5-20260909.safetensors",
}

CSV_FIELDS = [
    "benchmark",
    "method",
    "group",
    "pair_id",
    "score",
    "direction",
    "correct",
    "ground_truth",
    "runtime",
    "error",
    "indep_fw",
    "indep_rv",
]


def _csv_row(benchmark, method, group, pid, r):
    info = r["info"] or {}
    return {
        "benchmark": benchmark,
        "method": method,
        "group": group,
        "pair_id": pid,
        "score": r["score"],
        "direction": r["direction"],
        "correct": r["correct"],
        "ground_truth": r["ground_truth"],
        "runtime": r["runtime"],
        "error": r["error"],
        "indep_fw": info.get("indep_fw"),
        "indep_rv": info.get("indep_rv"),
    }


def _write_csv(csv_file, benchmark, method, results):
    with open(csv_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for group, group_results in results.items():
            for pid, r in group_results.items():
                writer.writerow(_csv_row(benchmark, method, group, pid, r))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark", required=True, choices=sorted(BENCHMARK_SETS)
    )
    parser.add_argument("--method", default="Loscar", choices=sorted(METHODS))
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
        default="results/bivariate",
        help=(
            "Folder in which the results should be stored, as "
            "'<method>_<benchmark>.pkl' and '.csv'."
        ),
    )
    parser.add_argument(
        "--ignore-pretraining-limits",
        action="store_true",
        help=(
            "TabPFN refuses to run on CPU with more than 1000 samples by "
            "default (a speed guard, not a capacity limit - it does not "
            "apply on CUDA). Pass this to override it, e.g. for the "
            "dataverse benchmark (n=1500) on CPU."
        ),
    )
    args = parser.parse_args()

    device = args.device or get_device()
    model = make_tabpfn_regressor(
        tabpfn_kwargs={
            "device": device,
            "model_path": MODEL_VERSIONS[args.model_version],
            "random_state": args.random_state,
            "ignore_pretraining_limits": args.ignore_pretraining_limits,
        }
    )
    run_method = METHODS[args.method]

    save_file = (
        Path(f"{args.results_directory}")
        / f"{args.method}_{args.benchmark}.pkl"
    )
    save_file.parent.mkdir(parents=True, exist_ok=True)
    csv_file = save_file.with_suffix(".csv")

    tasks = list(BENCHMARK_SETS[args.benchmark]())
    print(
        f"benchmark={args.benchmark} method={args.method} device={device} "
        f"n_pairs={len(tasks)}"
    )

    results = {}

    def finish_group(group, group_results):
        results[group] = group_results
        n_valid = sum(r["error"] is None for r in group_results.values())
        n_ok = sum(bool(r["correct"]) for r in group_results.values())
        if n_valid:
            print(
                f"{group} accuracy: {n_ok}/{n_valid} valid pairs "
                f"({n_ok / n_valid:.2%})"
            )
        else:
            print(f"{group}: no valid pairs")
        with open(save_file, "wb") as file:
            pickle.dump(results, file)
        _write_csv(csv_file, args.benchmark, args.method, results)

    current_group = None
    group_results = {}
    for i, (group, pid, build_dataset) in enumerate(tasks, start=1):
        if group != current_group:
            if current_group is not None:
                finish_group(current_group, group_results)
            current_group = group
            group_results = {}

        # synchronize CUDA so the timing only covers this pair
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        try:
            dataset = build_dataset()
            x = dataset.cause.flatten().numpy().reshape((-1, 1))
            y = dataset.effect.flatten().numpy().reshape((-1, 1))
            score, direction, info = run_method(x, y, args.random_state, model)
            correct = direction == GROUND_TRUTH
            error = None
        except Exception as exc:
            score, direction, info, correct = None, None, None, None
            error = repr(exc)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        runtime = time.time() - t0

        group_results[pid] = {
            "score": score,
            "direction": direction,
            "correct": correct,
            "ground_truth": GROUND_TRUTH,
            "info": info,
            "error": error,
            "runtime": runtime,
        }
        status = f"ERROR: {error}" if error else f"correct={correct}"
        print(
            f"[{group}] pair {pid} ({i}/{len(tasks)}): score={score} "
            f"{status} ({runtime:.1f}s)"
        )

    if current_group is not None:
        finish_group(current_group, group_results)


if __name__ == "__main__":
    main()
