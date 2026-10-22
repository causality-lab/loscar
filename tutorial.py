"""Tutorial: Running Loscar for graphs and cause-effect pairs."""

import numpy as np
import torch
from cdt.metrics import SHD
from sklearn.preprocessing import StandardScaler

from loscar.loscar import Loscar
from loscar.loscar_bivariate import Loscar_bivariate

from data.dgp import set_random_seed, simulate_dag, simulate_nonlinear_sem

SEED = 0
N_SAMPLES = 5000
# TabPFN runs on the GPU if one is available; set DEVICE = "cpu" to force CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running TabPFN on {DEVICE}")

set_random_seed(SEED)

# ---------------------------------------------------------------------------
# 1) Multivariate: Loscar on a simulated location-scale SEM with d=6 nodes
# ---------------------------------------------------------------------------
d = 6
B_true = simulate_dag(d=d, s0=d, graph_type="ER")  # B_true[i, j] = 1 means i -> j
X, _, _, _ = simulate_nonlinear_sem(N_SAMPLES, sem_type="LS-s", noise_dist="skew-t", het_noise_fraction=1.0, B=B_true)
X = StandardScaler().fit_transform(X)

model = Loscar(
    device=DEVICE,
    random_state=SEED,
    alpha=0.01,
    scale_quantiles=(0.05, 0.95),
    scale_method="quantile_range",
)
model.fit(X)
B_est = (model.adjacency_matrix_ != 0).astype(int)  # same convention as B_true

print("=== Loscar (d=6) ===")
print("True DAG:\n", B_true.astype(int))
print("Estimated DAG:\n", B_est)
print("Estimated causal order:", [int(i) for i in model.causal_order_])
print("SHD (cdt, reversed edge counts twice):", SHD(B_true, B_est, double_for_anticausal=True))

# ---------------------------------------------------------------------------
# 2) Bivariate: Loscar_bivariate on one cause-effect pair
# ---------------------------------------------------------------------------
B_pair = np.array([[0, 1], [0, 0]])
XY, _, _, _ = simulate_nonlinear_sem(N_SAMPLES, sem_type="LS-s", noise_dist="skew-t", het_noise_fraction=1.0, B=B_pair)
XY = StandardScaler().fit_transform(XY)
x, y = XY[:, :1], XY[:, 1:]
score, direction, res_fw, res_rv = Loscar_bivariate(x, y, device=DEVICE, random_state=SEED)

print("\n=== Loscar_bivariate ===")
print("score (> 0 favours x -> y):", score)
print("inferred direction:", "x -> y" if direction == 1 else "y -> x", "| truth: x -> y")
