"""
Lazy random walk binomial decomposition for DGPN.

Migrated from: lazy_random_walk_utils.py

Implements DGPN Eq. 13: binomial expansion of (beta*I + (1-beta)*A_hat)^K.
Each term i = C(K,i) * beta^{K-i} * ((1-beta)*A_hat)^i * X.

NOTE: This is NOT successive powers P^0 X, P^1 X, ..., P^K X.
Each part isolates a specific binomial term of the expansion.
"""

import numpy as np
import scipy.sparse as sp
from scipy.special import comb
import torch
from typing import List, Optional


def get_lazy_rw_ith_features(features, k, pos, A_hat, beta):
    """Compute i-th term of the binomial decomposition.

    res = C(k, pos) * beta^{k-pos} * ((1-beta)*A_hat)^{pos} * features
    """
    temp = np.power(beta, (k - pos))
    for i in range(pos):
        features = torch.spmm((1 - beta) * A_hat, features)
    return comb(k, pos) * temp * features


def get_lrw_feature_list(features, A_hat, k=3, beta=0.7):
    """Compute K+1 decomposed features via binomial expansion.

    Args:
        features: [N, D] node features
        A_hat: [N, N] pre-normalized sparse adjacency (torch sparse)
        k: number of hops (produces k+1 parts)
        beta: lazy probability (stay at self). beta>0.5 recommended.

    Returns:
        List of k+1 tensors, each [N, D].
    """
    featurelist = []
    for i in range(k + 1):
        temp = get_lazy_rw_ith_features(
            features.clone().detach(), k, i, A_hat, beta)
        featurelist.append(temp)
    return featurelist


# ---- Adjacency normalization ----

def normalize_adjacency_gcn(edge_index, num_nodes):
    """GCN renormalization: D_tilde^{-1/2} A_tilde D_tilde^{-1/2}."""
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    self_loops = np.arange(num_nodes)
    row = np.concatenate([row, self_loops])
    col = np.concatenate([col, self_loops])
    data = np.ones(len(row), dtype=np.float32)
    A_tilde = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    degrees = np.array(A_tilde.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return _sparse_scipy_to_torch(A_norm)


def normalize_adjacency_rw(edge_index, num_nodes):
    """Row-normalized: D^{-1} A."""
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    data = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    D_inv = sp.diags(1.0 / degrees)
    return _sparse_scipy_to_torch(D_inv @ A)


def _sparse_scipy_to_torch(mx):
    mx = mx.tocoo()
    indices = torch.LongTensor(np.vstack([mx.row, mx.col]))
    values = torch.FloatTensor(mx.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(mx.shape))


# ---- Convenience wrapper ----

def lazy_random_walk_decompose(edge_index, x, K=3, beta=0.7,
                               num_nodes=None, normalization="gcn"):
    """End-to-end: normalize adjacency + binomial decomposition.

    Args:
        edge_index: [2, E] graph connectivity
        x: [N, D] node features
        K: hops (default 3)
        beta: lazy probability (default 0.7)
        normalization: 'gcn' (symmetric) or 'rw' (row-normalized)

    Returns:
        List of K+1 tensors [N, D].
    """
    if num_nodes is None:
        num_nodes = x.size(0)
    norm_fn = {"gcn": normalize_adjacency_gcn, "rw": normalize_adjacency_rw}
    A_hat = norm_fn[normalization](edge_index, num_nodes)
    return get_lrw_feature_list(x, A_hat, K, beta)
