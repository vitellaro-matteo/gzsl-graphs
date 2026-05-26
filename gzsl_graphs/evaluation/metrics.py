"""
Evaluation metrics for ZSL and GZSL.

  I-ZSL: Top-1 accuracy on unseen nodes (search space = unseen classes only)
  GZSL-S: Accuracy on seen nodes (search space = all classes)
  GZSL-U: Accuracy on unseen nodes (search space = all classes)
  H: Harmonic mean = 2*S*U / (S+U)
"""

import torch
from typing import Dict, List


def compute_zsl_metrics(logits, labels, unseen_classes, test_mask) -> Dict[str, float]:
    """I-ZSL: accuracy on unseen test nodes, search restricted to unseen classes."""
    unseen_mask = torch.zeros_like(test_mask, dtype=torch.bool)
    for c in unseen_classes:
        unseen_mask |= (labels == c)
    mask = test_mask & unseen_mask

    if mask.sum() == 0:
        return {"i_zsl": 0.0}

    unseen_logits = logits[mask][:, unseen_classes]
    pred_idx = unseen_logits.argmax(dim=1)
    pred_classes = torch.tensor([unseen_classes[i] for i in pred_idx])
    acc = (pred_classes == labels[mask]).float().mean().item()
    return {"i_zsl": acc}


def compute_gzsl_metrics(logits, labels, seen_classes, unseen_classes,
                         test_mask) -> Dict[str, float]:
    """GZSL: Seen accuracy, Unseen accuracy, Harmonic mean."""
    preds = logits.argmax(dim=1)

    seen_mask = torch.zeros_like(test_mask, dtype=torch.bool)
    for c in seen_classes:
        seen_mask |= (labels == c)
    seen_test = test_mask & seen_mask
    s = (preds[seen_test] == labels[seen_test]).float().mean().item() if seen_test.sum() > 0 else 0.0

    unseen_mask = torch.zeros_like(test_mask, dtype=torch.bool)
    for c in unseen_classes:
        unseen_mask |= (labels == c)
    unseen_test = test_mask & unseen_mask
    u = (preds[unseen_test] == labels[unseen_test]).float().mean().item() if unseen_test.sum() > 0 else 0.0

    h = (2 * s * u) / (s + u) if (s + u) > 0 else 0.0
    return {"gzsl_s": s, "gzsl_u": u, "harmonic_mean": h}
