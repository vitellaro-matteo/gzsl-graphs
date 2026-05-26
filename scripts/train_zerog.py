#!/usr/bin/env python3
"""
ZeroG training: cross-dataset zero-shot transfer with GZSL evaluation.

Faithful to the original ZeroG repo (NineAbyss/ZeroG), with the addition
of GZSL metrics from our shared evaluation module.

Requires pre-processed .pt dataset files from:
  https://drive.google.com/drive/folders/1WfBIPA3dMd8qQZ6QlQRg9MIFGMwnPdFj

Usage:
    # In-domain transfer: train on citation, test on citation
    python scripts/train_zerog.py \
        --source Cora Citeseer Pubmed Arxiv \
        --target Cora Citeseer Pubmed \
        --data_dir ./data/zerog \
        --epochs 50 --k 2

    # Cross-domain: train on citation, test on co-purchase
    python scripts/train_zerog.py \
        --source Arxiv Cora Pubmed Citeseer \
        --target wikics \
        --data_dir ./data/zerog
"""

import argparse
import sys
import os
import time
import logging

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gzsl_graphs.models.zerog import ZeroG
from gzsl_graphs.data.zerog_subgraph import kHopSubgraphDataset
from gzsl_graphs.evaluation import compute_zsl_metrics, compute_gzsl_metrics
from gzsl_graphs.utils import set_seed, ExperimentLogger


def parse_args():
    p = argparse.ArgumentParser(description="ZeroG training")
    p.add_argument("--source", nargs="+", default=["Cora", "Citeseer", "Pubmed", "Arxiv"],
                   help="Source datasets for pre-training")
    p.add_argument("--target", nargs="+", default=["Cora", "Citeseer", "Pubmed"],
                   help="Target datasets for zero-shot evaluation")
    p.add_argument("--data_dir", default="./data/zerog",
                   help="Directory containing .pt dataset files")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--k", type=int, default=2, help="k-hop subgraph extraction")
    p.add_argument("--R", type=int, default=10, help="Neighbor aggregation rounds")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--if_norm", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output_dir", default="results/zerog/")
    return p.parse_args()


def load_zerog_dataset(data_dir, name):
    """Load a pre-processed .pt dataset file."""
    path = os.path.join(data_dir, f"{name.lower()}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ZeroG dataset not found: {path}\n"
            f"Run: python scripts/download_data.py --dataset zerog\n"
            f"Or download manually and place {name.lower()}.pt in {data_dir}/")
    data = torch.load(path, weights_only=False)
    # Ensure label_text attribute exists
    if not hasattr(data, "label_text"):
        data.label_text = data.label_name
    return data


class DataWrapper:
    """Minimal wrapper matching ZeroG's expected interface."""
    def __init__(self, data):
        self.data = data
        self.raw_texts = data.raw_texts
        self.label_text = data.label_text if hasattr(data, "label_text") else data.label_name


def evaluate_zerog(model, target_data, target_name, device, seen_classes=None,
                   unseen_classes=None):
    """Evaluate ZeroG on a target dataset.

    Returns dict with ZSL accuracy and optionally GZSL metrics.
    """
    model.eval()
    data = target_data.data if hasattr(target_data, "data") else target_data

    with torch.no_grad():
        logits = model.encode_graph(data, target_name)

    labels = data.y.long()
    if labels.dim() > 1:
        labels = labels.squeeze(1)

    # Test mask handling
    # ZeroG .pt files store test_mask as a list of tensors (multiple splits)
    # or a 2D tensor [N, num_splits]. Original code uses test_mask[0].
    if hasattr(data, "test_mask") and data.test_mask is not None:
        test_mask = data.test_mask
        if isinstance(test_mask, list):
            test_mask = test_mask[0]  # use first split (matches original)
        elif isinstance(test_mask, torch.Tensor) and test_mask.dim() > 1:
            test_mask = test_mask[:, 0]
        test_mask = test_mask.bool()
    else:
        test_mask = torch.ones(labels.size(0), dtype=torch.bool)

    # Standard ZSL: accuracy on all test nodes (all classes unseen in cross-dataset)
    preds = logits.argmax(dim=1).cpu()
    if test_mask.any():
        zsl_acc = (preds[test_mask] == labels[test_mask]).float().mean().item()
    else:
        zsl_acc = (preds == labels).float().mean().item()

    results = {"zsl_acc": zsl_acc, "dataset": target_name}

    # GZSL evaluation (if class splits provided)
    if seen_classes is not None and unseen_classes is not None:
        all_classes = sorted(set(seen_classes + unseen_classes))
        # Use all nodes as test for GZSL (ZeroG never trains on target)
        full_mask = torch.ones(labels.size(0), dtype=torch.bool)

        gzsl = compute_gzsl_metrics(
            logits.cpu(), labels, seen_classes, unseen_classes, full_mask)
        zsl_metrics = compute_zsl_metrics(
            logits.cpu(), labels, unseen_classes, full_mask)
        results.update(gzsl)
        results.update(zsl_metrics)

    return results


def get_gzsl_splits(target_name):
    """Get GZSL class splits for target datasets.

    Maps ZeroG's dataset naming convention (capitalized) to the canonical
    names used in our splits.py, then returns seen/unseen class lists
    from class_split_2.
    """
    from gzsl_graphs.data.splits import CLASS_SPLITS

    # Map ZeroG .pt file names -> our canonical dataset names
    name_map = {
        # Original datasets
        "Cora": "cora",
        "Citeseer": "citeseer",
        "Pubmed": "pubmed",
        "Arxiv": "ogbn-arxiv",
        # New datasets
        "WikiCS": "wikics",
        "wikics": "wikics",
        "AmazonComputers": "amazon-computers",
        "amazon-computers": "amazon-computers",
        "AmazonPhoto": "amazon-photo",
        "amazon-photo": "amazon-photo",
        "CoauthorCS": "coauthor-cs",
        "coauthor-cs": "coauthor-cs",
        "CoauthorPhysics": "coauthor-physics",
        "coauthor-physics": "coauthor-physics",
    }

    mapped = name_map.get(target_name)
    if mapped is None or mapped not in CLASS_SPLITS:
        return None, None

    split = CLASS_SPLITS[mapped].get("class_split_2", {})
    seen = split.get("train", []) + split.get("val", [])
    unseen = split.get("test", split.get("unseen", []))
    if not seen or not unseen:
        return None, None
    return seen, unseen


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # ---- Load target datasets for evaluation ----
    print("\nLoading target datasets...")
    targets = {}
    for name in args.target:
        data = load_zerog_dataset(args.data_dir, name)
        wrapper = DataWrapper(data)
        # Make edges undirected for Citeseer/Arxiv (matches original)
        if name in ["Citeseer", "Arxiv"]:
            wrapper.data.edge_index = to_undirected(wrapper.data.edge_index)
        targets[name] = wrapper
        n_nodes = wrapper.data.y.shape[0]
        n_classes = len(wrapper.data.y.unique())
        print(f"  {name}: {n_nodes} nodes, {n_classes} classes")

    # ---- Build training subgraph datasets ----
    print("\nBuilding training subgraphs from source datasets...")
    train_datasets = []
    for name in args.source:
        data = load_zerog_dataset(args.data_dir, name)
        if name in ["Citeseer", "Arxiv"]:
            data.edge_index = to_undirected(data.edge_index)
        if not hasattr(data, "label_text"):
            data.label_text = data.label_name

        # Arxiv uses 1-hop (too large for k-hop), others use args.k
        hops = 1 if name == "Arxiv" else args.k
        sg_dataset = kHopSubgraphDataset(
            data, num_hops=hops, max_nodes=100, dataset_name=name)
        train_datasets.append(sg_dataset)

    concat_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(concat_dataset, batch_size=args.batch_size,
                              shuffle=True)
    print(f"Total training subgraphs: {len(concat_dataset)}")
    print(f"Training batches per epoch: {len(train_loader)}")

    # ---- Initialize model ----
    model = ZeroG(R=args.R, if_norm=args.if_norm, device=device).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {trainable:,} trainable / {total:,} total params")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=1.0)

    logger = ExperimentLogger(args.output_dir)

    # ---- Pre-training evaluation (epoch 0) ----
    print("\n" + "=" * 70)
    print("Pre-training evaluation (epoch 0)")
    print("=" * 70)
    for name, wrapper in targets.items():
        seen, unseen = get_gzsl_splits(name)
        res = evaluate_zerog(model, wrapper, name, device, seen, unseen)
        print(f"  {name}: ZSL={res['zsl_acc']:.4f}", end="")
        if "harmonic_mean" in res:
            print(f"  S={res['gzsl_s']:.4f} U={res['gzsl_u']:.4f} H={res['harmonic_mean']:.4f}", end="")
        print()

    # ---- Training loop ----
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = 0
        start = time.time()

        for step, batch in enumerate(train_loader):
            data = batch[0].to(device)
            loss = model(data)

            if torch.isnan(loss).any():
                print(f"  NaN loss at step {step}, skipping")
                continue

            loss = loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum
            n_steps += 1

        elapsed = time.time() - start
        avg_loss = epoch_loss / max(n_steps, 1)
        print(f"\nEpoch {epoch}: loss={avg_loss:.4f} ({elapsed:.1f}s)")

        # ---- Evaluate on all targets ----
        for name, wrapper in targets.items():
            seen, unseen = get_gzsl_splits(name)
            res = evaluate_zerog(model, wrapper, name, device, seen, unseen)
            print(f"  {name}: ZSL={res['zsl_acc']:.4f}", end="")
            if "harmonic_mean" in res:
                print(f"  S={res['gzsl_s']:.4f} U={res['gzsl_u']:.4f} "
                      f"H={res['harmonic_mean']:.4f}", end="")
            print()
            logger.log(epoch=epoch, model="zerog", **res)

    logger.save()
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()