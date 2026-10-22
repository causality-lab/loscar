"""Utility functions for simulating data and evaluating causal discovery algorithms.

`simulate_dag` is adapted from the `notears` package (https://github.com/xunzheng/notears), 
file notears/utils.py, by Xun Zheng, Bryon Aragam, Pradeep Ravikumar and Eric P. Xing. 
Licensed under the Apache License 2.0, see LICENSES/notears-Apache-2.0.txt.
"""

import torch
import numpy as np
import pandas as pd
import igraph as ig
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from skewt_scipy.skewt import skewt

def set_random_seed(seed):
    """Set random seed for reproducibility across random, numpy, and torch (incl. cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        """Randomly permute the rows and columns of M."""
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        """Randomly orient edges of undirected graph B_und to make it acyclic."""
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        """Convert igraph graph to adjacency matrix."""
        return np.array(G.get_adjacency().data)

    if graph_type == "ER":
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == "BP":
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    elif graph_type == "Fully":
        B = np.triu(np.ones((d, d)), 1)
    else:
        raise ValueError("unknown graph type")
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def sample_nonlin_fct(mech_type, X_parents):
    """Sample mean function for nonlinear SEM."""
    def _single_sigmoids(X_pa_j):
        """Sample sigmoidal function for nonlinear SEM."""
        c = random.uniform(-2, 2)
        bern = np.random.binomial(1, 0.5)
        b = bern * random.uniform(0.5, 2) + (1 - bern) * random.uniform(-2, -0.5)
        a = np.random.exponential(scale=1 / 4) + 2.5  # changed from +1 to +2.5
        return a * (b * (X_pa_j + c)) / (1 + abs(b * (X_pa_j + c)))
    if mech_type == "gp":
        gp = GaussianProcessRegressor()  # RBF kernel with length_scale=1
        x_child = gp.sample_y(X_parents, random_state=None).flatten()
    elif mech_type == "sigmoidal":
        sigmoids_list = []
        for j in range(X_parents.shape[1]):
            sigmoids_list.append(_single_sigmoids(X_parents[:, j]))
        x_child = np.sum(sigmoids_list, axis=0)
    else:
        raise ValueError("unknown mech type")
    return x_child


def sample_noise(noise_type, noise_dist, n, x_child=None):
    """Sample noise function for nonlinear SEM."""
    if noise_type == "additive":
        noise = sample_noise_dist(noise_dist, n, static=True) * 0.2
    elif noise_type == "LS":
        assert x_child is not None
        noise = sample_noise_dist(noise_dist, n, static=True) * 0.2
        noise = (x_child - min(x_child)) * noise * 1.75
    else:
        raise ValueError("unknown noise type")
    return noise


def select_het_noise_nodes(G, ordered_vertices, het_noise_fraction):
    """Select nodes to be heteroscedastic noise nodes.

    Args:
        G (ig.Graph): graph
        ordered_vertices (list): ordered vertices
        het_noise_fraction (float): fraction of non-source nodes exhibiting het noise

    Returns:
       node_het_noise_flag (dict): {node: het_noise_flag}
    """
    if not 0 <= het_noise_fraction <= 1:
        raise ValueError("het_noise_fraction must be between 0 and 1")
    non_source_node_flg = [len(G.neighbors(j, mode=ig.IN)) != 0 for j in ordered_vertices]
    nodes_df = pd.DataFrame(
        {
            "ordered_vertices": ordered_vertices,
            "non_source_node_flg": non_source_node_flg,
        }
    )
    nbr_non_source_nodes = nodes_df["non_source_node_flg"].sum()
    nbr_het_noise_nodes = int(np.ceil(het_noise_fraction * nbr_non_source_nodes))

    het_noise_nodes = np.random.choice(
        nodes_df[nodes_df["non_source_node_flg"]]["ordered_vertices"],
        size=nbr_het_noise_nodes,
        replace=False,
    )

    nodes_df["het_noise_node"] = nodes_df["ordered_vertices"].isin(het_noise_nodes)
    return nodes_df.set_index("ordered_vertices")["het_noise_node"].to_dict()


def sample_noise_dist(noise_dist, n, static=False):
    """Sample from noise distribution."""
    if noise_dist == "gauss":
        if static:
            sigma = 1
        else:
            sigma = np.random.uniform(low=1, high=np.sqrt(2), size=n)
        x_child = np.random.normal(loc=0, scale=sigma, size=n)
    elif noise_dist == "skew-t":
        alpha = -2
        df = 4
        loc = 0
        if static:
            scale = 0.912871
            x_child = skewt.rvs(a=alpha, df=df, loc=loc, scale=scale, size=n)
        else:
            scale = np.random.uniform(low=0.912871, high=1.290994, size=n)
            x_child = np.array([
                skewt.rvs(a=alpha, df=df, loc=loc, scale=s, size=1)[0] for s in scale
            ])
    elif noise_dist == "exponential":
        if static:
            lmbd = 1
        else:
            lmbd = np.random.uniform(low=1/np.sqrt(2), high=1, size=n)
        x_child = np.random.exponential(scale=lmbd, size=n)
    elif noise_dist == "coinflip":
        noise_list = ["gauss", "skew-t", "exponential"]
        choice = np.random.randint(0, len(noise_list))
        x_child = sample_noise_dist(noise_list[choice], n, static)
    else:
        raise ValueError(f"unknown noise distribution: {noise_dist}")
    return x_child


def simulate_single_equation(
    X,
    parents,
    noise_dist,
    sem_type,
    het_noise_node,
    B_cause_mean=None,
    B_cause_var=None,
):
    """
    Simulate a single equation of nonlinear SEM.

    Args:
        X (np.array): [n, d]
        parents (list): list of parents
        noise_dist (str): 'gauss', 'uniform', 'exponential', 'coinflip'
        sem_type (str): One of 'AN', 'AN-s', 'MN', 'LS', 'LS-s','LS-p'
        het_noise_node (bool): True if this node is affected by heteroscedastic noise
        B_cause_mean (np.array): [d, d] track which parents cause mean, only relevant for LS-p
        B_cause_var (np.array): [d, d] track which parent cause variance, only relevant for LS-p
    Returns:
        sim_function (np.array): [n, ] returns mean + noise
        sim_mean_noise (np.array): [n, 2] returns mean, noise
    """
    X_parents = X[:, parents]
    n, pa_size = X_parents.shape

    if pa_size == 0:
        # source node in DAG
        noise = np.zeros(n)
        x_child = sample_noise_dist(noise_dist, n)
    else:
        # non-source node in DAG
        if sem_type == "AN":
            x_child = sample_nonlin_fct(mech_type="gp", X_parents=X_parents)
            noise = sample_noise(noise_type="additive", noise_dist=noise_dist, n=n)
        elif sem_type == "AN-s":
            x_child = sample_nonlin_fct(mech_type="sigmoidal", X_parents=X_parents)
            noise = sample_noise(noise_type="additive", noise_dist=noise_dist, n=n)
        elif sem_type == "LS":
            x_child = sample_nonlin_fct(mech_type="gp", X_parents=X_parents)
            if het_noise_node:
                noise = sample_noise(noise_type="LS", noise_dist=noise_dist, n=n, x_child=x_child)
            else:
                noise = sample_noise(noise_type="additive", noise_dist=noise_dist, n=n)
        elif sem_type == "LS-s":
            x_child = sample_nonlin_fct(mech_type="sigmoidal", X_parents=X_parents)
            if het_noise_node:
                noise = sample_noise(noise_type="LS", noise_dist=noise_dist, n=n, x_child=x_child)
            else:
                noise = sample_noise(noise_type="additive", noise_dist=noise_dist, n=n)
        elif sem_type == "MN-s":
            noise_function = sample_nonlin_fct(mech_type="sigmoidal", X_parents=X_parents)
            noise = noise_function * np.random.uniform(low=0, high=1, size=n)
            x_child = np.zeros(n)
        elif sem_type == "LS-p":
            assert B_cause_mean is not None, "B_cause_mean must be provided for LS-p"
            assert B_cause_var is not None, "B_cause_var must be provided for LS-p"
            if pa_size == 1:
                # if only one parent, then this parents influences the mean and noise
                _, sim_mean_noise = simulate_single_equation(
                    X, parents, noise_dist, sem_type="LS", het_noise_node=1
                )
                x_child = sim_mean_noise[:, 0]
                noise = sim_mean_noise[:, 1]
                B_cause_mean[parents] = 1
                B_cause_var[parents] = 1
            else:
                # if more than one parent, then select half of the parents
                # that only affect mean and half that only affect noise
                nbr_PA_affect_noise = int(np.ceil(0.5 * pa_size))
                PA_affect_noise = np.random.choice(parents, size=nbr_PA_affect_noise, replace=False)
                PA_affect_mean = np.array(list(set(parents) - set(PA_affect_noise)))
                # print('PA_affect_noise', PA_affect_noise)
                # print('PA_affect_mean', PA_affect_mean)

                B_cause_mean[PA_affect_mean] = 1
                B_cause_var[PA_affect_noise] = 1

                x_child = sample_nonlin_fct(mech_type="gp", X_parents=X[:, PA_affect_mean])
                noise_function = sample_nonlin_fct(mech_type="sigmoidal", X_parents=X[:, PA_affect_noise])
                positive_noise_function = (noise_function - noise_function.min()) * 1.5
                noise = positive_noise_function * sample_noise(
                    noise_type="additive", noise_dist=noise_dist, n=n
                )
        else:
            raise ValueError("unknown sem type")

    sim_function = x_child + noise
    sim_mean_noise = np.stack((x_child, noise), axis=1)
    return sim_function, sim_mean_noise


def simulate_nonlinear_sem(n, sem_type="AN", noise_dist="gauss", het_noise_fraction=0, d=None, B=None, s0=None):
    """Simulate samples from nonlinear SEM.

    Either `B` or `d` must be given.
      - If `B` is given, `d` defaults to `B.shape[1]`. If both are given and they
        disagree, a ValueError is raised.
      - If `B` is not given, it is generated from an Erdos-Renyi graph using `d` and
        `s0` (s0 defaults to `d`, i.e. an expected out-degree of 1 per node).

    Args:
        n (int): num of samples
        sem_type (str): One of 'AN', 'AN-s', 'MN', 'LS', 'LS-s', 'LS-p'
        noise_dist (str): 'gauss', 'skew-t', 'exponential', 'coinflip'
        het_noise_fraction (float): fraction of none source nodes that are affected with het-noise
        d (int): num of nodes; required if B is not given, else defaults to B.shape[1]
        B (np.ndarray): [d, d] binary adj matrix of DAG; generated from an ER graph if not given
        s0 (int): expected num of edges when generating B from d; defaults to d
    Returns:
        X (np.ndarray): [n, d] sample matrix
        X_inspect (np.ndarray): [n, d, 2] sampled mean and noise functions separately
        B_cause_mean: [d, d] adjacency matrix with respect to mean function only
        B_cause_var: [d, d] adjacency matrix with respect to variance function only
    """
    if B is None and d is None:
        raise ValueError("either B or d must be given")
    if B is not None:
        if d is None:
            d = B.shape[1]
        elif B.shape[1] != d:
            raise ValueError(
                f"B and d do not match: B.shape[1]={B.shape[1]} but d={d}. "
                "When B is given, d defaults to B.shape[1]."
            )
    else:
        s0 = d if s0 is None else s0
        B = simulate_dag(d=d, s0=s0, graph_type="ER")

    X = np.zeros([n, d])
    X_inspect = np.zeros([n, d, 2])  # store mean and noise functions separately
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    het_noise_flag = select_het_noise_nodes(G, ordered_vertices, het_noise_fraction)
    if sem_type == "LS-p":
        B_cause_mean = np.zeros((d, d))  # track which parents cause mean
        B_cause_var = np.zeros((d, d))  # track which parents cause variance
    else:
        B_cause_mean = None
        B_cause_var = None
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        if sem_type == "LS-p":
            X[:, j], X_inspect[:, j] = simulate_single_equation(
                X,
                parents,
                noise_dist=noise_dist,
                sem_type=sem_type,
                het_noise_node=het_noise_flag[j],
                B_cause_mean=B_cause_mean[:, j],
                B_cause_var=B_cause_var[:, j],
            )
        else:
            X[:, j], X_inspect[:, j] = simulate_single_equation(
                X,
                parents,
                noise_dist=noise_dist,
                sem_type=sem_type,
                het_noise_node=het_noise_flag[j],
            )
    return X, X_inspect, B_cause_mean, B_cause_var