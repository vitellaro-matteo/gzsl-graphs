#!/usr/bin/env python3
"""
Transductive baseline for graph GZSL (thesis Chapter 7).

Three phases:
  1. Train 2-layer GraphSAGE on full graph (transductive), CE on seen nodes
  2. Train dual projectors (NCE + alignment, thesis Eq. 7.15-7.17)
  3. Constrained clustering: fix seen centroids, refine unseen from semantics

Usage:
    python scripts/train_baseline.py --dataset cora --data_root ./data
    python scripts/train_baseline.py --dataset cora --data_root ./data --seeds 0 1 2 3 4
"""

import argparse
import sys
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gzsl_graphs.data import GraphZSLDataset
from gzsl_graphs.models.transductive_baseline import (
    GraphFeatureEnhancer, GraphProjector, SemanticProjector,
    constrained_clustering,
)
from gzsl_graphs.evaluation import compute_zsl_metrics, compute_gzsl_metrics
from gzsl_graphs.utils import set_seed, ExperimentLogger


def parse_args():
    p = argparse.ArgumentParser(description="Transductive baseline (thesis Ch.7)")
    p.add_argument("--dataset", default="cora")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    # Phase 1: GNN
    p.add_argument("--enhanced_dim", type=int, default=256)
    p.add_argument("--epochs_phase1", type=int, default=200)
    p.add_argument("--lr_phase1", type=float, default=0.01)
    # Phase 2: Dual projectors
    p.add_argument("--joint_dim", type=int, default=512)
    p.add_argument("--epochs_phase2", type=int, default=150)
    p.add_argument("--lr_phase2", type=float, default=1e-4)
    p.add_argument("--lambda_align", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=512)
    # Phase 3: Clustering
    p.add_argument("--max_iterations", type=int, default=20)
    # General
    p.add_argument("--output_dir", default="results/baseline/")
    return p.parse_args()


# ============================================================
# Phase 1: Train GraphSAGE
# ============================================================

def train_phase1(data, args, device):
    """Train GraphSAGE on full graph, cross-entropy on seen nodes."""
    print("\n" + "=" * 60)
    print("Phase 1: GraphSAGE feature enhancement")
    print("=" * 60)

    model = GraphFeatureEnhancer(
        in_features=data.feature_dim,
        hidden_features=args.enhanced_dim,
        out_features=args.enhanced_dim,
        num_layers=2, dropout=0.5,
    ).to(device)

    classifier = nn.Linear(args.enhanced_dim, len(data.seen_classes)).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr_phase1, weight_decay=5e-4)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    labels = data.y.to(device)
    train_mask = data.train_mask.to(device)

    # Map labels to [0, num_seen)
    label_map = {c: i for i, c in enumerate(data.seen_classes)}
    train_labels = labels[train_mask].cpu()
    mapped = torch.tensor([label_map[l.item()] for l in train_labels]).to(device)

    best_acc = 0
    for epoch in range(args.epochs_phase1):
        model.train(); classifier.train()
        optimizer.zero_grad()
        enhanced = model(x, edge_index)
        loss = F.cross_entropy(classifier(enhanced[train_mask]), mapped)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == args.epochs_phase1 - 1:
            model.eval()
            with torch.no_grad():
                pred = classifier(model(x, edge_index)[train_mask]).argmax(1)
                acc = (pred == mapped).float().mean().item()
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f} acc={acc:.4f}")
            best_acc = max(best_acc, acc)

    print(f"  Best train acc: {best_acc:.4f}")

    model.eval()
    with torch.no_grad():
        enhanced = model(x, edge_index)
    return model, enhanced


# ============================================================
# Phase 2: Train dual projectors (thesis Eq. 7.15-7.17)
# ============================================================

def train_phase2(enhanced, data, args, device):
    """Train dual projectors with NCE + alignment loss."""
    print("\n" + "=" * 60)
    print("Phase 2: Dual space projection")
    print("=" * 60)

    graph_dim = enhanced.size(1)
    bert_dim = data.class_semantics.size(1)

    graph_proj = GraphProjector(graph_dim, args.joint_dim, hidden_dim=1024).to(device)
    semantic_proj = SemanticProjector(bert_dim, args.joint_dim, hidden_dim=768).to(device)

    optimizer = optim.AdamW(
        list(graph_proj.parameters()) + list(semantic_proj.parameters()),
        lr=args.lr_phase2, weight_decay=1e-5)

    # Training data - index on CPU, then move to device
    train_features = enhanced[data.train_mask].detach().to(device)
    train_labels = data.y[data.train_mask].to(device)

    # Seen class attributes (L2-normalized)
    seen_attrs = F.normalize(
        data.class_semantics[data.seen_classes].to(device), p=2, dim=1)

    # Label mapping
    label_map = {c: i for i, c in enumerate(data.seen_classes)}
    train_labels_local = torch.tensor(
        [label_map[l.item()] for l in train_labels.cpu()], device=device)

    num_train = train_features.size(0)
    num_seen = len(data.seen_classes)

    for epoch in range(args.epochs_phase2):
        graph_proj.train(); semantic_proj.train()
        epoch_loss = 0; n_batches = 0

        perm = torch.randperm(num_train, device=device)
        for i in range(0, num_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch_feats = train_features[idx]
            batch_labels = train_labels_local[idx]

            # Project to joint space
            graph_joint = graph_proj(batch_feats)         # [B, joint_dim]
            bert_joint = semantic_proj(seen_attrs)         # [C_seen, joint_dim]

            # NCE loss (thesis Eq. 7.15)
            logits = graph_joint @ bert_joint.T / args.temperature
            nce_loss = F.cross_entropy(logits, batch_labels)

            # Alignment loss (thesis Eq. 7.16): per-class centroid alignment
            align_loss = torch.tensor(0.0, device=device)
            for c_local in range(num_seen):
                c_mask = (batch_labels == c_local)
                if c_mask.sum() == 0:
                    continue
                class_mean = graph_joint[c_mask].mean(dim=0, keepdim=True)
                target = bert_joint[c_local:c_local + 1]
                align_loss = align_loss + (class_mean - target).pow(2).sum()
            align_loss = align_loss / max(num_seen, 1)

            # Combined loss (thesis Eq. 7.17)
            loss = nce_loss + args.lambda_align * align_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 20 == 0 or epoch == args.epochs_phase2 - 1:
            graph_proj.eval(); semantic_proj.eval()
            with torch.no_grad():
                sample = train_features[:min(1000, num_train)]
                sample_labels = train_labels_local[:min(1000, num_train)]
                g = graph_proj(sample)
                b = semantic_proj(seen_attrs)
                acc = (g @ b.T).argmax(1).eq(sample_labels).float().mean().item()
            print(f"  Epoch {epoch:3d}: loss={epoch_loss/n_batches:.4f} align_acc={acc:.4f}")

    graph_proj.eval(); semantic_proj.eval()
    return graph_proj, semantic_proj


# ============================================================
# Phase 3: Clustering + Evaluation
# ============================================================

def evaluate_phase3(enhanced, data, graph_proj, semantic_proj, args, device):
    """Constrained clustering + GZSL evaluation."""
    print("\n" + "=" * 60)
    print("Phase 3: Constrained clustering + evaluation")
    print("=" * 60)

    # Move enhanced to device, keep masks on CPU for indexing
    enhanced_dev = enhanced.to(device)
    labels = data.y

    # Compute seen centroids (thesis Eq. 7.18): mean of training node projections
    with torch.no_grad():
        train_proj = graph_proj(enhanced_dev[data.train_mask.to(device)])
        train_proj = F.normalize(train_proj, p=2, dim=1)

    train_labels = labels[data.train_mask].to(device)
    seen_centroids = []
    for c in data.seen_classes:
        mask = (train_labels == c)
        if mask.sum() > 0:
            centroid = train_proj[mask].mean(dim=0)
            centroid = F.normalize(centroid, p=2, dim=0)
        else:
            centroid = torch.zeros(args.joint_dim, device=device)
        seen_centroids.append(centroid)
    seen_centroids = torch.stack(seen_centroids)

    # Initialize unseen centroids from semantic projections (thesis Eq. 7.19)
    with torch.no_grad():
        unseen_attrs = F.normalize(
            data.class_semantics[data.unseen_classes].to(device), p=2, dim=1)
        unseen_centroids_init = semantic_proj(unseen_attrs)

    # Run constrained clustering
    test_mask = data.test_seen_mask | data.test_unseen_mask

    predictions, confidences, unseen_centroids = constrained_clustering(
        graph_proj, enhanced_dev, test_mask.to(device),
        seen_centroids, unseen_centroids_init,
        max_iterations=args.max_iterations)

    # Map predictions back to global class IDs
    all_classes = list(data.seen_classes) + list(data.unseen_classes)
    test_labels = labels[test_mask].cpu()
    pred_global = torch.tensor([all_classes[p.item()] for p in predictions.cpu()])

    # Build logits-like tensor for our shared evaluation
    # Create a similarity matrix [N_test, C] that our metrics can consume
    num_test = test_mask.sum().item()
    num_classes = data.num_classes
    logits = torch.full((num_test, num_classes), -1e9)

    with torch.no_grad():
        test_proj = graph_proj(enhanced_dev[test_mask.to(device)])
        test_proj = F.normalize(test_proj, p=2, dim=1)
        all_centroids = torch.cat([seen_centroids, unseen_centroids], dim=0)

        sims = test_proj @ all_centroids.T  # [N_test, num_seen + num_unseen]

    # Place similarities at correct class indices
    for i, c in enumerate(data.seen_classes):
        logits[:, c] = sims[:, i].cpu()
    for i, c in enumerate(data.unseen_classes):
        logits[:, len(data.seen_classes) + i] = sims[:, len(data.seen_classes) + i].cpu()
        # Actually map to correct column
        logits[:, c] = sims[:, len(data.seen_classes) + i].cpu()

    # Shared evaluation
    test_labels_full = labels[test_mask].cpu()
    test_mask_all = torch.ones(num_test, dtype=torch.bool)

    zsl = compute_zsl_metrics(logits, test_labels_full, data.unseen_classes, test_mask_all)
    gzsl = compute_gzsl_metrics(logits, test_labels_full, data.seen_classes,
                                data.unseen_classes, test_mask_all)

    return {**zsl, **gzsl}


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = []

    for seed in args.seeds:
        print(f"\n{'#' * 60}")
        print(f"  Seed {seed}")
        print(f"{'#' * 60}")
        set_seed(seed)

        data = GraphZSLDataset(args.dataset, root=args.data_root,
                               random_seed=seed).load()

        # Phase 1
        gnn, enhanced = train_phase1(data, args, device)

        # Phase 2
        graph_proj, semantic_proj = train_phase2(enhanced, data, args, device)

        # Phase 3 + eval
        results = evaluate_phase3(enhanced, data, graph_proj, semantic_proj, args, device)

        print(f"\n  I-ZSL={results['i_zsl']:.4f}  S={results['gzsl_s']:.4f}  "
              f"U={results['gzsl_u']:.4f}  H={results['harmonic_mean']:.4f}")
        all_results.append(results)

        # Cleanup
        del gnn, enhanced, graph_proj, semantic_proj
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Aggregate
    print("\n" + "=" * 60)
    print(f"AGGREGATE RESULTS ({len(all_results)} seeds)")
    print("=" * 60)

    for key in ["i_zsl", "gzsl_s", "gzsl_u", "harmonic_mean"]:
        vals = [r[key] for r in all_results]
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"  {key:20s}: {mean:.4f} +/- {std:.4f}")

    logger = ExperimentLogger(args.output_dir)
    for i, r in enumerate(all_results):
        logger.log(model="transductive_baseline", dataset=args.dataset,
                   seed=args.seeds[i], **r)
    logger.save()
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()