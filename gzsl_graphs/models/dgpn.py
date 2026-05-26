"""
Decomposed Graph Prototype Network (DGPN).

Wang et al., KDD 2021. https://github.com/zhengwang100/dgpn

Architecture (faithful to original implementation):
  - Single shared linear (fc1) encodes each decomposed subpart
  - Local head (fc_local_pred_csd) maps each part to semantic space
  - Global head (fc_final_pred_csd) maps the summed embedding
  - Similarity via dot product against class semantic descriptions
  - Compositional pooling: simple SUM (not weighted omega_k from paper)

This file contains ONLY the model. Data loading, transforms, evaluation elsewhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def dot_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Dot-product similarity. Replaces util_functions.dot_sim."""
    return torch.mm(x, y)


class DGPN(nn.Module):
    """Decomposed Graph Prototype Network.

    Args:
        input_dim: Node feature dimension (1433 Cora, 128 ogbn-arXiv, etc.)
        hidden_dim: Must match CSD dimension (384 for MiniLM, 128 for BERT-Tiny)
        dropout: Dropout probability (default: 0.5)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_local_pred_csd = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc_final_pred_csd = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.dropout = dropout
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, feature_list: List[torch.Tensor],
                csd_matrix: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            feature_list: K+1 tensors [N, input_dim] from decomposition
            csd_matrix: [C, hidden_dim] class semantic descriptions

        Returns:
            preds: [N, C] global similarity scores
            local_preds: list of K+1 [N, C] per-part scores
        """
        device = next(self.parameters()).device
        part_embeddings = []
        local_preds = []

        for features in feature_list:
            h = self.fc1(features.to(device))
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            part_embeddings.append(h)
            local_sem = self.fc_local_pred_csd(h)
            local_preds.append(dot_sim(local_sem, csd_matrix.t()))

        global_embedding = torch.sum(torch.stack(part_embeddings), dim=0)
        global_sem = self.fc_final_pred_csd(global_embedding)
        global_sem = F.dropout(global_sem, p=self.dropout, training=self.training)
        preds = dot_sim(global_sem, csd_matrix.t())

        return preds, local_preds


def compute_dgpn_loss(preds, local_preds, labels, seen_classes,
                      alpha=1.0, beta=0.7):
    """Joint loss: Q = Q_com + alpha * Q_loc.

    Args:
        preds: [N, C] global predictions
        local_preds: list of K+1 [N, C] local predictions
        labels: [N] ground truth (original class IDs)
        seen_classes: list of seen class indices
        alpha: local loss weight
        beta: (unused here, kept for API consistency)
    """
    seen_set = set(seen_classes)
    seen_idx = sorted(seen_set)
    label_map = {c: i for i, c in enumerate(seen_idx)}

    mask = torch.tensor([l.item() in seen_set for l in labels], dtype=torch.bool)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=preds.device)

    mapped = torch.tensor([label_map[l.item()] for l in labels[mask]],
                          device=preds.device, dtype=torch.long)

    loss_com = F.cross_entropy(preds[mask][:, seen_idx], mapped)

    loss_loc = torch.tensor(0.0, device=preds.device)
    for lp in local_preds:
        loss_loc += F.cross_entropy(lp[mask][:, seen_idx], mapped)
    loss_loc /= len(local_preds)

    return loss_com + alpha * loss_loc