"""Loaders for the real-world multivariate datasets.

Datasets
--------
- SACHS: 11 continuous protein-signaling variables, real flow-cytometry
  observational data (Sachs et al. 2005). Expected in data/sachs/.
- LightTunnel: Observational data set `lt_interventions_standard_v1` from
  CausalChambers project (Gamella et al. 2025). Downloaded to
  data/causal_chambers/ on first use (requires the `causalchamber` package).
  The data is licensed under the Creative Commons Attribution 4.0
  International License, see LICENSES/causal-chambers-4.0.txt.

References
----------
Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005).
    Causal protein-signaling networks derived from multiparameter single-cell
    data. Science, 308(5721), 523-529. https://doi.org/10.1126/science.1105809
Gamella, J. L., Peters, J., & Bühlmann, P. (2025). Causal chambers as a
    real-world physical testbed for AI methodology. Nature Machine
    Intelligence, 7, 107-118. https://doi.org/10.1038/s42256-024-00964-x
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

RNG_SEED = 0
N_SUBSAMPLE = (
    None  # None = use the full dataset (all samples), as in the paper
)

SACHS_DIR = os.path.join(DATA_DIR, "sachs")
SACHS_GT_NPY = os.path.join(SACHS_DIR, "sachs_ground_truth.npy")
SACHS_OBS_CSV = os.path.join(SACHS_DIR, "sachs_observational.csv")


def load_sachs(n_sub=N_SUBSAMPLE, seed=RNG_SEED):
    """SACHS"""
    true_adj = np.load(SACHS_GT_NPY).astype(int)

    df = pd.read_csv(SACHS_OBS_CSV)
    var_names = list(df.columns)
    X = df.to_numpy()
    if n_sub is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n_sub, replace=False)
        X = X[idx]
    return X, true_adj, var_names


LT_VARS = [
    "red",
    "green",
    "blue",
    "current",
    "ir_1",
    "ir_2",
    "ir_3",
    "vis_1",
    "vis_2",
    "vis_3",
    "pol_1",
    "pol_2",
    "angle_1",
    "angle_2",
    "l_11",
    "l_12",
    "l_21",
    "l_22",
    "l_31",
    "l_32",
]


def load_light_tunnel():
    "Full `uniform_reference` experiment (n=10000)"
    from causalchamber.datasets import Dataset
    from causalchamber import ground_truth

    root = os.path.join(DATA_DIR, "causal_chambers")
    os.makedirs(root, exist_ok=True)
    ds = Dataset("lt_interventions_standard_v1", root=root, download=True)
    exp = ds.get_experiment("uniform_reference")
    X = exp.as_pandas_dataframe()[LT_VARS].values
    true_adj = (
        ground_truth.graph("lt", "standard").loc[LT_VARS, LT_VARS].values
    )
    return X, true_adj, LT_VARS


DATASETS = {
    "SACHS": load_sachs,
    "LightTunnel": load_light_tunnel,
}
