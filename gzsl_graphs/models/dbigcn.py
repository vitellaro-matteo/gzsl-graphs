"""
Dual Bidirectional Graph Convolutional Networks (DBiGCN).

Yue et al., KDD 2022. https://github.com/warmerspring/DBiGCN

Architecture (faithful to original implementation):
  - BiGCN_X (node perspective): 2-layer MLP (d->512->C), wrapped in
    bidirectional propagation: Y_X = dropout(fc2(relu(fc1(S_V @ X))) @ S_A)
    Output: [N, C] — predictions from node perspective
  - BiGCN_A (class perspective): 1-layer MLP (d_csd->N), wrapped in
    bidirectional propagation: Y_A = dropout(fc1(S_A_norm @ CSD) @ S_V)
    Output: [C, N] — predictions from class perspective
  - Loss: L = L_nodes + alpha * L_classes + beta * 1e-6 * L_consistency

NOTE: The original repo uses BiGCN (1-layer) for the class branch, NOT the
BiGCN_A class (2-layer) that is also defined in BiGCN.py. This file follows
the actual main_cora.py training script.

This file contains ONLY the model + loss. Data loading and evaluation elsewhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional


class DBiGCN_X(nn.Module):
    """BiGCN from perspective of nodes (Eq. 4 in paper).

    Forward: Y_V = dropout(fc2(relu(fc1(S_V @ X))) @ S_A)

    Args:
        input_dim: Node feature dimension
        num_classes: Total number of classes (seen + unseen)
        hidden_dim: Hidden layer size (default: 512, matches original)
        dropout: Dropout probability
    """

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, X: torch.Tensor, S_V: torch.Tensor,
                S_A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Node features [N, D]
            S_V: Normalized node adjacency [N, N]
            S_A: Normalized class adjacency [C, C]

        Returns:
            Y_X: [N, C] prediction logits from node perspective
        """
        features = torch.mm(S_V, X)          # [N, D] graph propagation
        features = self.fc1(features)          # [N, hidden]
        features = self.act(features)          # [N, hidden]
        features = self.fc2(features)          # [N, C]
        Y_X = torch.mm(features, S_A)         # [N, C] class propagation
        Y_X = F.dropout(Y_X, p=self.dropout, training=self.training)
        return Y_X


class DBiGCN_A(nn.Module):
    """BiGCN from perspective of classes (Eq. 6 in paper).

    Forward: Y_A = dropout(fc1(S_A_norm @ CSD) @ S_V)

    NOTE: Original repo uses single-layer BiGCN here (not the 2-layer BiGCN_A
    class that is also defined in the file but unused in main_cora.py).

    Args:
        csd_dim: Dimension of class semantic descriptions
        num_nodes: Total number of nodes in the graph
        dropout: Dropout probability
    """

    def __init__(self, csd_dim: int, num_nodes: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(csd_dim, num_nodes, bias=True)
        self.dropout = dropout
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, CSD: torch.Tensor, S_A: torch.Tensor,
                S_V: torch.Tensor) -> torch.Tensor:
        """
        Args:
            CSD: Class semantic descriptions [C, D_csd]
            S_A: Normalized class adjacency [C, C]
            S_V: Normalized node adjacency [N, N]

        Returns:
            Y_A: [C, N] prediction logits from class perspective
        """
        features = torch.mm(S_A, CSD)         # [C, D_csd] class propagation
        features = self.fc1(features)          # [C, N]
        Y_A = torch.mm(features, S_V)         # [C, N] node propagation
        Y_A = F.dropout(Y_A, p=self.dropout, training=self.training)
        return Y_A


class DBiGCN(nn.Module):
    """Dual Bidirectional GCN combining node and class perspectives.

    Wraps BiGCN_X and BiGCN_A into a single module for convenience.

    Args:
        input_dim: Node feature dimension
        csd_dim: CSD embedding dimension
        num_classes: Total classes (seen + unseen)
        num_nodes: Total nodes in graph
        hidden_dim: Hidden size for BiGCN_X (default: 512)
        dropout: Dropout probability
    """

    def __init__(self, input_dim: int, csd_dim: int, num_classes: int,
                 num_nodes: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.model_X = DBiGCN_X(input_dim, num_classes, hidden_dim, dropout)
        self.model_A = DBiGCN_A(csd_dim, num_nodes, dropout)
        self.num_classes = num_classes

    def forward(self, X: torch.Tensor, CSD: torch.Tensor,
                S_V: torch.Tensor, S_A: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X: Node features [N, D]
            CSD: Class semantic descriptions [C, D_csd]
            S_V: Normalized node adjacency [N, N]
            S_A: Normalized class adjacency [C, C]

        Returns:
            Y_X: [N, C] predictions from node perspective
            Y_A: [C, N] predictions from class perspective
        """
        Y_X = self.model_X(X, S_V, S_A)
        Y_A = self.model_A(CSD, S_A, S_V)
        return Y_X, Y_A


def compute_dbigcn_loss(
    Y_X: torch.Tensor,
    Y_A: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    num_classes: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compute DBiGCN loss: L = L_nodes + alpha * L_classes + beta * 1e-6 * L_consistency.

    Args:
        Y_X: [N, C] raw logits from node perspective
        Y_A: [C, N] raw logits from class perspective
        labels: [N] ground truth class indices
        train_idx: Indices of training (seen-class) nodes
        num_classes: Total number of classes
        alpha: Weight for class-perspective loss
        beta: Weight for consistency loss (scaled by 1e-6 internally)

    Returns:
        Scalar loss.
    """
    device = Y_X.device

    # L_nodes: cross-entropy from node perspective
    loss_nodes = F.cross_entropy(Y_X[train_idx], labels[train_idx])

    # L_classes: cross-entropy from class perspective (Y_A is [C, N], transpose)
    loss_classes = F.cross_entropy(Y_A.T[train_idx], labels[train_idx])

    # L_consistency: ||softmax(Y_X_L) @ softmax(Y_A_L) - Y_true @ Y_true^T||_F^2
    num_train = len(train_idx)
    Y_true = torch.zeros(num_train, num_classes, device=device)
    Y_true[torch.arange(num_train), labels[train_idx]] = 1.0

    Y_X_soft = F.softmax(Y_X[train_idx], dim=1)     # [L, C]
    Y_A_soft = F.softmax(Y_A[:, train_idx], dim=0)   # [C, L]

    diff = torch.mm(Y_X_soft, Y_A_soft) - torch.mm(Y_true, Y_true.T)
    loss_consistency = torch.norm(diff, p="fro") ** 2

    return loss_nodes + alpha * loss_classes + beta * 1e-6 * loss_consistency


# ============================================================
# Class adjacency construction (from util_functions.py)
# ============================================================

def build_class_adjacency(
    csd_matrix: torch.Tensor,
    n_neighbors: int = 2,
    normalize: bool = True,
) -> torch.Tensor:
    """Build class adjacency matrix from CSD k-nearest neighbors.

    Computes pairwise distances between class embeddings, finds k-NN,
    creates binary adjacency, then symmetrically normalizes.

    Args:
        csd_matrix: [C, D] class semantic descriptions
        n_neighbors: Number of nearest neighbors (default: 2)
        normalize: Whether to apply symmetric normalization

    Returns:
        [C, C] dense tensor (normalized class adjacency S_A)
    """
    data = csd_matrix.detach().cpu().numpy()

    # Pairwise squared distances
    sum_x = np.sum(np.square(data), axis=1)
    dist = np.add(np.add(-2 * np.dot(data, data.T), sum_x).T, sum_x)

    n = dist.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        # k nearest neighbors (excluding self)
        neighbors = np.argsort(dist[i])[1:1 + n_neighbors]
        W[i, neighbors] = 1
        W[neighbors, i] = 1  # make symmetric

    if normalize:
        W = _symmetric_normalize(W)

    return torch.tensor(W, dtype=torch.float32)


def build_node_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Build node adjacency from edge_index, symmetrically normalize.

    Args:
        edge_index: [2, E] COO edges
        num_nodes: Number of nodes
        normalize: Whether to apply symmetric normalization

    Returns:
        [N, N] dense tensor (normalized node adjacency S_V)
    """
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    adj = sp.coo_matrix(
        (np.ones(len(row)), (row, col)),
        shape=(num_nodes, num_nodes), dtype=np.float32,
    )
    # Symmetrize
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if normalize:
        adj = _symmetric_normalize(adj.toarray())
    else:
        adj = adj.toarray()

    return torch.tensor(adj, dtype=torch.float32)


def _symmetric_normalize(adj):
    """D^{-1/2} A D^{-1/2} normalization. Input: numpy array or sparse."""
    adj_sp = sp.coo_matrix(adj)
    rowsum = np.array(adj_sp.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)
    return np.array(D @ adj_sp @ D.T.todense())