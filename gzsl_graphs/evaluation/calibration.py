"""
Post-hoc calibration for GZSL evaluation.

Migrated from: calibration_utils.py

  1. Temperature scaling: logits / T
  2. Bias correction (calibrated stacking): logits_seen -= gamma

Includes grid search and heatmap visualization.

NOT CALLED DURING TRAINING. JUST HERE FOR COMPLETIONS SAKE.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from .metrics import compute_gzsl_metrics


def temperature_scale_logits(logits: np.ndarray, T: float) -> np.ndarray:
    """Scale logits by temperature T. logits: (N, C) numpy array."""
    if T == 1.0:
        return logits
    return logits / float(T)


def bias_correct_logits(logits: np.ndarray, seen_class_indices, gamma: float) -> np.ndarray:
    """Subtract gamma from seen-class logits."""
    if gamma == 0 or len(seen_class_indices) == 0:
        return logits
    logits = logits.copy()
    logits[:, seen_class_indices] -= float(gamma)
    return logits


def apply_calibration(logits: np.ndarray, T: float, gamma: float,
                      seen_class_indices) -> np.ndarray:
    """Apply temperature scaling then bias correction."""
    l = temperature_scale_logits(logits, T)
    return bias_correct_logits(l, seen_class_indices, gamma)


def calibration_grid_search(logits, labels, seen_classes, unseen_classes,
                            test_mask, temperature_range=None, gamma_range=None) -> Dict:
    """Grid search (T, gamma) for best GZSL harmonic mean. Accepts numpy or torch."""
    if temperature_range is None:
        temperature_range = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    if gamma_range is None:
        gamma_range = [0, 1, 2, 3, 5, 7, 10]

    logits_np = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.array(logits)
    labels_t = labels.cpu() if isinstance(labels, torch.Tensor) else torch.LongTensor(labels)
    mask_t = test_mask.cpu() if isinstance(test_mask, torch.Tensor) else torch.BoolTensor(test_mask)

    grid = []
    heatmap = np.zeros((len(temperature_range), len(gamma_range)))
    best_h, best_params, best_metrics = -1.0, (1.0, 0.0), {}

    for i, T in enumerate(temperature_range):
        for j, gamma in enumerate(gamma_range):
            cal = apply_calibration(logits_np, T, gamma, seen_classes)
            metrics = compute_gzsl_metrics(torch.FloatTensor(cal), labels_t,
                                           seen_classes, unseen_classes, mask_t)
            h = metrics["harmonic_mean"]
            heatmap[i, j] = h
            grid.append({"T": T, "gamma": gamma, **metrics})
            if h > best_h:
                best_h, best_params, best_metrics = h, (T, gamma), metrics

    return {"best_T": best_params[0], "best_gamma": best_params[1],
            "best_H": best_h, "best_S": best_metrics.get("gzsl_s", 0.0),
            "best_U": best_metrics.get("gzsl_u", 0.0),
            "grid": grid, "heatmap": heatmap,
            "T_range": temperature_range, "gamma_range": gamma_range}


def plot_calibration_heatmap(result, dataset_name="", save_path=None):
    """Plot harmonic mean heatmap across (T, gamma) grid."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(result["heatmap"], aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(result["gamma_range"])))
    ax.set_xticklabels([str(g) for g in result["gamma_range"]])
    ax.set_yticks(range(len(result["T_range"])))
    ax.set_yticklabels([str(t) for t in result["T_range"]])
    ax.set_xlabel("gamma (seen penalty)")
    ax.set_ylabel("Temperature T")
    title = f"Harmonic mean across (T, gamma) grid"
    if dataset_name: title += f", {dataset_name}"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Harmonic mean")
    best_ti = result["T_range"].index(result["best_T"])
    best_gi = result["gamma_range"].index(result["best_gamma"])
    ax.plot(best_gi, best_ti, "r*", markersize=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
