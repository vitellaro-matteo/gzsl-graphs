#!/usr/bin/env python3
"""
ICIS training: Image-free Classifier Injection with Semantics, adapted for graphs.

Christensen et al., ICCV 2023, adapted to graph data (thesis Chapter 5).

Pipeline:
  1. Load graph data via shared GraphZSLDataset
  2. Train base MLP classifier on seen-class features
  3. Train ICIS joint autoencoder (attribute <-> weight spaces)
  4. Predict unseen class weights from semantic descriptions
  5. Inject weights into extended classifier
  6. Evaluate GZSL with shared metrics

Usage:
    python scripts/train_icis.py --dataset cora --data_root ./data
    python scripts/train_icis.py --dataset ogbn-arxiv --data_root ./data --epochs 1000
"""

import argparse
import copy
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gzsl_graphs.data import GraphZSLDataset
from gzsl_graphs.models.icis import (
    Autoencoder, JointAutoencoder, LinearClassifier, MLPClassifier, weights_init,
)
from gzsl_graphs.models.icis_adapter import ICISDataAdapter
from gzsl_graphs.evaluation import compute_zsl_metrics, compute_gzsl_metrics
from gzsl_graphs.utils import set_seed, ExperimentLogger


def parse_args():
    p = argparse.ArgumentParser(description="ICIS for graph GZSL")
    p.add_argument("--dataset", default="cora")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--seed", type=int, default=42)
    # Base classifier
    p.add_argument("--classifier_epochs", type=int, default=100)
    p.add_argument("--classifier_lr", type=float, default=0.0001)
    p.add_argument("--classifier_hidden", type=int, default=256)
    # ICIS joint autoencoder
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=1000)
    p.add_argument("--num_layers", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--cos_sim_loss", action="store_true", default=False)
    # Weight normalization (ICIS-specific)
    p.add_argument("--wn_factor", type=float, default=10.0,
                   help="Weight normalization for unseen weights (higher=stronger)")
    # General
    p.add_argument("--output_dir", default="results/icis/")
    p.add_argument("--patience", type=int, default=50,
                   help="Early stopping patience (0=disabled)")
    return p.parse_args()


# ============================================================
# Phase 1: Train base classifier
# ============================================================

def train_base_classifier(adapter, args, device):
    """Train linear base classifier on seen-class graph features.

    ICIS requires a LINEAR classifier (not MLP) because the extracted
    weights [W, b] must have the same dimensionality as the input features.
    This is fundamental to ICIS: predicted unseen weights are injected
    into the same classifier architecture.
    """
    print("\n" + "=" * 60)
    print("Phase 1: Training base classifier on seen classes")
    print("=" * 60)

    feature_dim = adapter.feature_dim
    num_seen = adapter.num_seen

    model = LinearClassifier(feature_dim, num_seen)
    model.apply(weights_init)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.classifier_lr)
    criterion = nn.CrossEntropyLoss()

    # Map labels to [0, num_seen)
    train_X = adapter.train_feature.to(device)
    train_Y = _map_labels(adapter.train_label, adapter.seenclasses).to(device)

    best_acc = 0
    best_model = None

    for epoch in range(args.classifier_epochs):
        model.train()
        perm = torch.randperm(train_X.size(0))
        epoch_loss = 0
        n_batches = 0

        for i in range(0, train_X.size(0), args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = model(train_X[idx])
            loss = criterion(logits, train_Y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 20 == 0 or epoch == args.classifier_epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(train_X)
                acc = (logits.argmax(1) == train_Y).float().mean().item()
            print(f"  Epoch {epoch:3d}: loss={epoch_loss/n_batches:.4f} acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(model)

    print(f"  Best training accuracy: {best_acc:.4f}")

    # Extract classifier weights [W, b] — shape [num_seen, feature_dim + 1]
    weight = best_model.fc.weight.data.cpu()
    bias = best_model.fc.bias.data.cpu()
    target_weights = torch.cat([weight, bias.unsqueeze(1)], dim=1)
    print(f"  Target weights shape: {target_weights.shape}")
    print(f"  Expected: [{num_seen}, {feature_dim} + 1 = {feature_dim + 1}]")
    print(f"  Mean weight norm: {target_weights.norm(dim=1).mean():.4f}")

    return best_model, target_weights


# ============================================================
# Phase 2: Train ICIS joint autoencoder
# ============================================================

def train_icis(adapter, target_weights, args, device):
    """Train ICIS joint autoencoder for weight prediction."""
    print("\n" + "=" * 60)
    print("Phase 2: Training ICIS joint autoencoder")
    print("=" * 60)

    attr_dim = adapter.attribute.size(1)
    weight_dim = target_weights.size(1)

    # Build joint autoencoder
    ae_attr = Autoencoder(attr_dim, args.embed_dim, num_layers=args.num_layers)
    ae_weight = Autoencoder(weight_dim, args.embed_dim, num_layers=args.num_layers)
    model = JointAutoencoder(ae_attr, ae_weight)
    model.apply(weights_init)
    model.to(device)

    # Training data: (seen_attributes, seen_classifier_weights)
    seen_attrs = adapter.attribute[adapter.seenclasses].to(device)
    seen_weights = target_weights.to(device)

    dataset = TensorDataset(seen_attrs, seen_weights)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    if args.cos_sim_loss:
        criterion = _cos_sim_loss
    else:
        criterion = nn.MSELoss(reduction="none")

    best_H = 0
    best_model = None
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for batch_attrs, batch_weights in loader:
            optimizer.zero_grad()
            output = model((batch_attrs, batch_weights))
            (att_from_att, att_from_weight, weight_from_weight,
             weight_from_att, latent_att, latent_weight) = output

            # 4 reconstruction losses (ICIS Eq. 4-5 + cross-reconstruction)
            l_aa = criterion(att_from_att, batch_attrs).mean()
            l_wa = criterion(att_from_weight, batch_attrs).mean()
            l_ww = criterion(weight_from_weight, batch_weights).mean()
            l_aw = criterion(weight_from_att, batch_weights).mean()

            loss = l_aa + l_wa + l_ww + l_aw

            # Magnitude guidance (small weight, from your joint_latent.py)
            target_mag = seen_weights.norm(dim=1).mean().detach()
            pred_norms = weight_from_att.norm(dim=1)
            mag_loss = torch.abs(pred_norms - target_mag).mean()
            loss = loss + 0.05 * mag_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(len(loader), 1)

        # Evaluate periodically
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            model.eval()
            results = predict_and_evaluate(
                model, adapter, target_weights, args, device)
            H = results["harmonic_mean"]
            zsl = results["i_zsl"]

            print(f"  Epoch {epoch:4d}: loss={epoch_loss:.6f} "
                  f"I-ZSL={zsl:.4f} S={results['gzsl_s']:.4f} "
                  f"U={results['gzsl_u']:.4f} H={H:.4f}")

            if H >= best_H:
                best_H = H
                best_model = copy.deepcopy(model)
                no_improve = 0
            else:
                no_improve += 1

            if args.patience > 0 and no_improve >= args.patience // 50:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  Best H: {best_H:.4f}")
    return best_model


# ============================================================
# Phase 3: Predict unseen weights + evaluate
# ============================================================

@torch.no_grad()
def predict_and_evaluate(model, adapter, target_weights, args, device):
    """Predict unseen class weights and evaluate GZSL."""
    model.eval()
    feature_dim = adapter.feature_dim

    # Predict unseen class weights: A -> Z -> W
    unseen_attrs = adapter.attribute[adapter.unseenclasses].to(device)
    predicted_weights = model.predict(unseen_attrs)

    # Build extended classifier [seen_weights; predicted_unseen_weights]
    num_seen = adapter.num_seen
    num_unseen = adapter.num_unseen
    num_total = num_seen + num_unseen

    ext_model = LinearClassifier(feature_dim, num_total).to(device)

    # Inject seen class weights (from trained base classifier)
    tw = target_weights.to(device)
    ext_model.fc.weight.data[:num_seen] = tw[:, :-1]
    ext_model.fc.bias.data[:num_seen] = tw[:, -1]

    # Inject predicted unseen class weights
    pw = predicted_weights
    ext_model.fc.weight.data[num_seen:] = pw[:, :feature_dim]
    ext_model.fc.bias.data[num_seen:] = pw[:, feature_dim]

    # Weight normalization (ICIS-specific)
    if args.wn_factor > 0:
        unseen_w = torch.cat([
            ext_model.fc.weight.data[num_seen:],
            ext_model.fc.bias.data[num_seen:].unsqueeze(1)
        ], dim=1)
        unseen_w = unseen_w / (args.wn_factor * unseen_w.norm(dim=1).mean())
        ext_model.fc.weight.data[num_seen:] = unseen_w[:, :feature_dim]
        ext_model.fc.bias.data[num_seen:] = unseen_w[:, feature_dim]

    # Get logits for all test nodes
    ext_model.eval()
    all_features = torch.cat([
        adapter.test_seen_feature,
        adapter.test_unseen_feature
    ]).to(device)
    all_logits = ext_model(all_features).cpu()

    # Build labels in extended space: seen=[0..num_seen), unseen=[num_seen..num_total)
    seen_labels_mapped = _map_labels(adapter.test_seen_label, adapter.seenclasses)
    unseen_labels_mapped = _map_labels_extend(
        adapter.test_unseen_label, adapter.unseenclasses, adapter.seenclasses)
    all_labels = torch.cat([seen_labels_mapped, unseen_labels_mapped])

    # Use our shared evaluation
    seen_cls = list(range(num_seen))
    unseen_cls = list(range(num_seen, num_total))
    test_mask = torch.ones(all_labels.size(0), dtype=torch.bool)

    gzsl = compute_gzsl_metrics(all_logits, all_labels, seen_cls, unseen_cls, test_mask)

    # I-ZSL: unseen-only model on unseen test features
    unseen_model = LinearClassifier(feature_dim, num_unseen).to(device)
    unseen_model.fc.weight.data[:] = pw[:, :feature_dim]
    unseen_model.fc.bias.data[:] = pw[:, feature_dim]
    if args.wn_factor > 0:
        unseen_model.fc.weight.data[:] = unseen_w[:, :feature_dim]
        unseen_model.fc.bias.data[:] = unseen_w[:, feature_dim]

    unseen_logits = unseen_model(adapter.test_unseen_feature.to(device)).cpu()
    unseen_labels_zsl = _map_labels(adapter.test_unseen_label, adapter.unseenclasses)
    unseen_mask = torch.ones(unseen_labels_zsl.size(0), dtype=torch.bool)
    zsl = compute_zsl_metrics(
        unseen_logits, unseen_labels_zsl,
        list(range(num_unseen)), unseen_mask)

    results = {**gzsl, **zsl}
    return results


# ============================================================
# Helpers
# ============================================================

def _map_labels(labels, classes):
    """Map original class labels to [0, len(classes))."""
    mapped = torch.LongTensor(labels.size())
    for i in range(classes.size(0)):
        mapped[labels == classes[i]] = i
    return mapped


def _map_labels_extend(labels, unseen_classes, seen_classes):
    """Map unseen labels to [len(seen), len(seen)+len(unseen))."""
    mapped = torch.LongTensor(labels.size())
    for i in range(unseen_classes.size(0)):
        mapped[labels == unseen_classes[i]] = i + seen_classes.size(0)
    return mapped


def _cos_sim_loss(pred, target):
    """Cosine similarity loss (1 - cos_sim)."""
    pred_n = torch.nn.functional.normalize(pred, p=2, dim=-1)
    target_n = torch.nn.functional.normalize(target, p=2, dim=-1)
    return (1 - (pred_n * target_n).sum(dim=-1)).unsqueeze(-1)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data via shared loader
    data = GraphZSLDataset(args.dataset, root=args.data_root,
                           random_seed=args.seed).load()

    # Adapt to ICIS format
    adapter = ICISDataAdapter(data)
    print(f"\n  Dataset: {args.dataset}")
    print(f"  Features: {adapter.feature_dim}")
    print(f"  Seen classes: {adapter.num_seen} {adapter.seenclasses.tolist()}")
    print(f"  Unseen classes: {adapter.num_unseen} {adapter.unseenclasses.tolist()}")
    print(f"  Attribute dim: {adapter.attribute.size(1)}")
    print(f"  Train: {adapter.train_feature.size(0)} nodes")
    print(f"  Test seen: {adapter.test_seen_feature.size(0)} nodes")
    print(f"  Test unseen: {adapter.test_unseen_feature.size(0)} nodes")

    # Phase 1: Train base classifier
    base_model, target_weights = train_base_classifier(adapter, args, device)

    # Phase 2: Train ICIS
    best_model = train_icis(adapter, target_weights, args, device)

    # Phase 3: Final evaluation
    print("\n" + "=" * 60)
    print("Final evaluation")
    print("=" * 60)
    results = predict_and_evaluate(best_model, adapter, target_weights, args, device)
    print(f"\n  {args.dataset}:")
    print(f"  I-ZSL = {results['i_zsl']:.4f}")
    print(f"  GZSL-S = {results['gzsl_s']:.4f}")
    print(f"  GZSL-U = {results['gzsl_u']:.4f}")
    print(f"  H = {results['harmonic_mean']:.4f}")

    # Save
    logger = ExperimentLogger(args.output_dir)
    logger.log(model="icis", dataset=args.dataset, seed=args.seed, **results)
    logger.save()
    print(f"\n  Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()