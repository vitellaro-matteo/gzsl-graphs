"""
Transductive Baseline for Graph GZSL (thesis Chapter 7).

Three-phase approach:
  Phase 1: GraphSAGE feature enhancement (2-layer, transductive)
  Phase 2: Dual space projection (NCE + alignment loss)
  Phase 3: Constrained clustering (fixed seen centroids, refined unseen)

Architecture:
  GraphSAGE: in_features -> 256-dim enhanced features
  Graph projector: 256 -> 1024 (hidden) -> 512 (joint), with residual
  Semantic projector: 384 -> 768 (hidden) -> 512 (joint), with residual
  Clustering: cosine similarity, 20 iterations or convergence

This file contains model components only. Training in scripts/train_baseline.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Phase 1: GraphSAGE Feature Enhancement
# ============================================================

class SimpleSAGEConv(nn.Module):
    """GraphSAGE convolution (pure PyTorch, no torch-geometric needed)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin_self = nn.Linear(in_features, out_features, bias=False)
        self.lin_neigh = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)

    def forward(self, x, edge_index):
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        out_self = self.lin_self(x)

        # Mean neighbor aggregation
        out_neigh = torch.zeros(N, x.size(1), device=x.device, dtype=x.dtype)
        degree = torch.zeros(N, device=x.device, dtype=x.dtype)
        out_neigh.index_add_(0, dst, x[src])
        degree.index_add_(0, dst, torch.ones(src.size(0), device=x.device))
        out_neigh = out_neigh / degree.clamp(min=1).unsqueeze(1)
        out_neigh = self.lin_neigh(out_neigh)

        return out_self + out_neigh + self.bias


class GraphFeatureEnhancer(nn.Module):
    """2-layer GraphSAGE for feature enhancement (thesis Eq. 7.1)."""

    def __init__(self, in_features, hidden_features=256, out_features=256,
                 dropout=0.5, num_layers=2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SimpleSAGEConv(in_features, hidden_features))
        for _ in range(num_layers - 2):
            self.convs.append(SimpleSAGEConv(hidden_features, hidden_features))
        if num_layers > 1:
            self.convs.append(SimpleSAGEConv(hidden_features, out_features))

        for i in range(num_layers):
            dim = hidden_features if i < num_layers - 1 else out_features
            self.bns.append(nn.BatchNorm1d(dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ============================================================
# Phase 2: Dual Space Projectors
# ============================================================

class GraphProjector(nn.Module):
    """Project graph features to joint space (thesis Eq. 7.9-7.11).

    Architecture: input -> hidden (1024) -> 2 residual blocks -> output (512)
    With skip connection from input to output.
    """

    def __init__(self, graph_dim=256, joint_dim=512, hidden_dim=1024):
        super().__init__()
        self.input_proj = nn.Linear(graph_dim, hidden_dim)
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(0.2))
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(0.2))
        self.output_head = nn.Linear(hidden_dim, joint_dim)
        self.skip = nn.Linear(graph_dim, joint_dim)

    def forward(self, x):
        h = self.input_proj(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return F.normalize(self.output_head(h) + self.skip(x), p=2, dim=1)


class SemanticProjector(nn.Module):
    """Project BERT embeddings to joint space (thesis Eq. 7.12-7.14).

    Architecture: input -> hidden (768) -> 2 residual blocks -> output (512)
    With skip connection. Lighter dropout (BERT already well-regularized).
    """

    def __init__(self, bert_dim=384, joint_dim=512, hidden_dim=768):
        super().__init__()
        self.input_proj = nn.Linear(bert_dim, hidden_dim)
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(0.1))
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(0.1))
        self.output_head = nn.Linear(hidden_dim, joint_dim)
        self.skip = nn.Linear(bert_dim, joint_dim)

    def forward(self, x):
        h = self.input_proj(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return F.normalize(self.output_head(h) + self.skip(x), p=2, dim=1)


# ============================================================
# Phase 3: Constrained Clustering
# ============================================================

@torch.no_grad()
def constrained_clustering(
    graph_proj, test_features, test_mask,
    seen_centroids, unseen_centroids_init,
    max_iterations=20, convergence_threshold=0.001,
):
    """Constrained clustering (thesis Section 7.4).

    Seen centroids are FIXED. Unseen centroids are iteratively refined
    by averaging assigned test node projections.

    Args:
        graph_proj: Trained graph projector
        test_features: [N, enhanced_dim] all node enhanced features
        test_mask: [N] boolean mask for test nodes
        seen_centroids: [num_seen, joint_dim] fixed
        unseen_centroids_init: [num_unseen, joint_dim] from semantic projector

    Returns:
        predictions: [N_test] predicted class indices (0..num_seen+num_unseen-1)
        confidences: [N_test] cosine similarities
        unseen_centroids: [num_unseen, joint_dim] refined centroids
    """
    graph_proj.eval()

    # Project test nodes to joint space
    test_joint = graph_proj(test_features[test_mask])
    test_joint = F.normalize(test_joint, p=2, dim=1)

    unseen_centroids = unseen_centroids_init.clone()
    num_seen = seen_centroids.size(0)
    num_unseen = unseen_centroids.size(0)

    for iteration in range(max_iterations):
        all_centroids = torch.cat([seen_centroids, unseen_centroids], dim=0)

        # Assign by cosine similarity (thesis Eq. 7.20)
        similarities = test_joint @ all_centroids.T
        assignments = similarities.argmax(dim=1)

        # Update unseen centroids only (thesis Eq. 7.21)
        max_change = 0.0
        new_unseen = []
        for i in range(num_unseen):
            mask = (assignments == num_seen + i)
            if mask.sum() == 0:
                new_unseen.append(unseen_centroids[i])
                continue
            new_c = test_joint[mask].mean(dim=0)
            new_c = F.normalize(new_c, p=2, dim=0)
            max_change = max(max_change, (new_c - unseen_centroids[i]).norm().item())
            new_unseen.append(new_c)

        unseen_centroids = torch.stack(new_unseen)

        if max_change < convergence_threshold:
            break

    # Final assignment
    all_centroids = torch.cat([seen_centroids, unseen_centroids], dim=0)
    similarities = test_joint @ all_centroids.T
    predictions = similarities.argmax(dim=1)
    confidences = similarities.max(dim=1).values

    return predictions, confidences, unseen_centroids