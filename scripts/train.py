#!/usr/bin/env python3
"""
Train any graph ZSL model on any dataset.

Usage:
    python scripts/train.py --model dgpn --dataset cora --setting gzsl
    python scripts/train.py --model dbigcn --dataset cora --setting gzsl
    python scripts/train.py --config configs/dgpn.yaml
"""

import argparse, yaml, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from gzsl_graphs.data import GraphZSLDataset
from gzsl_graphs.evaluation import compute_zsl_metrics, compute_gzsl_metrics
from gzsl_graphs.utils import ExperimentLogger, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train graph ZSL model")
    p.add_argument("--model", default="dgpn", choices=["dgpn", "dbigcn"])
    p.add_argument("--dataset", default="cora")
    p.add_argument("--setting", default="gzsl", choices=["zsl", "gzsl"])
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--output_dir", default="results/")
    p.add_argument("--data_root", default="./data")
    # DGPN-specific
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--beta", type=float, default=0.7)
    # Shared loss weight
    p.add_argument("--alpha", type=float, default=1.0)
    # DBiGCN-specific
    p.add_argument("--wd", type=float, default=0.0001, help="Weight decay")
    p.add_argument("--loss_beta", type=float, default=1.0,
                   help="DBiGCN consistency loss weight")
    p.add_argument("--hidden_dim", type=int, default=512,
                   help="DBiGCN BiGCN_X hidden dim")
    p.add_argument("--n_neighbors", type=int, default=2,
                   help="DBiGCN class adjacency k-NN")
    return p.parse_args()


# ============================================================
# DGPN training
# ============================================================

def train_dgpn(args, data, device):
    from gzsl_graphs.models.dgpn import DGPN, compute_dgpn_loss
    from gzsl_graphs.transforms import lazy_random_walk_decompose

    print(f"Decomposing (K={args.K}, beta={args.beta})...")
    decomposed = lazy_random_walk_decompose(
        data.edge_index, data.x, K=args.K, beta=args.beta)
    decomposed = [z.to(device) for z in decomposed]

    semantic_dim = data.class_semantics.size(1)
    model = DGPN(input_dim=data.feature_dim, hidden_dim=semantic_dim,
                 dropout=args.dropout).to(device)

    csd = data.class_semantics.to(device)
    labels = data.y.to(device)
    train_mask = data.train_mask.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def forward_fn():
        preds, local_preds = model(decomposed, csd)
        return preds, local_preds

    def loss_fn(preds, aux):
        local_preds = aux
        return compute_dgpn_loss(
            preds[train_mask], [lp[train_mask] for lp in local_preds],
            labels[train_mask], data.seen_classes, alpha=args.alpha)

    def get_logits(preds, aux):
        return preds  # Y_X is the prediction

    return model, optimizer, forward_fn, loss_fn, get_logits


# ============================================================
# DBiGCN training
# ============================================================

def train_dbigcn(args, data, device):
    from gzsl_graphs.models.dbigcn import (
        DBiGCN, compute_dbigcn_loss,
        build_class_adjacency, build_node_adjacency,
    )

    # Build adjacency matrices
    print("Building node adjacency S_V...")
    S_V = build_node_adjacency(data.edge_index, data.n_nodes).to(device)
    print(f"  S_V: {S_V.shape}")

    print(f"Building class adjacency S_A (k={args.n_neighbors} NN)...")
    S_A = build_class_adjacency(data.class_semantics,
                                n_neighbors=args.n_neighbors).to(device)
    print(f"  S_A: {S_A.shape}")

    model = DBiGCN(
        input_dim=data.feature_dim,
        csd_dim=data.class_semantics.size(1),
        num_classes=data.num_classes,
        num_nodes=data.n_nodes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    X = data.x.to(device)
    csd = data.class_semantics.to(device)
    labels = data.y.to(device)
    train_idx = torch.where(data.train_mask)[0].to(device)

    # Original uses separate optimizers for X and A branches
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.wd)

    def forward_fn():
        Y_X, Y_A = model(X, csd, S_V, S_A)
        return Y_X, Y_A

    def loss_fn(preds, aux):
        Y_X, Y_A = preds, aux
        return compute_dbigcn_loss(
            Y_X, Y_A, labels, train_idx,
            num_classes=data.num_classes,
            alpha=args.alpha, beta=args.loss_beta)

    def get_logits(preds, aux):
        return preds  # Y_X is the prediction (Eq. 10)

    return model, optimizer, forward_fn, loss_fn, get_logits


# ============================================================
# Shared training loop
# ============================================================

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = GraphZSLDataset(args.dataset, root=args.data_root,
                           random_seed=args.seed).load()

    # Model-specific setup
    if args.model == "dgpn":
        model, optimizer, forward_fn, loss_fn, get_logits = \
            train_dgpn(args, data, device)
    elif args.model == "dbigcn":
        model, optimizer, forward_fn, loss_fn, get_logits = \
            train_dbigcn(args, data, device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Model: {args.model}, {sum(p.numel() for p in model.parameters()):,} params")

    logger = ExperimentLogger(args.output_dir)
    best_val, patience_ctr, best_state = -1, 0, None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        preds, aux = forward_fn()
        loss = loss_fn(preds, aux)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                preds_eval, aux_eval = forward_fn()
            logits = get_logits(preds_eval, aux_eval).cpu()
            tm = data.test_seen_mask | data.test_unseen_mask

            if args.setting == "gzsl":
                m = compute_gzsl_metrics(logits, data.y, data.seen_classes,
                                         data.unseen_classes, tm)
                val = m["harmonic_mean"]
                s = f"S={m['gzsl_s']:.4f} U={m['gzsl_u']:.4f} H={val:.4f}"
            else:
                m = compute_zsl_metrics(logits, data.y, data.unseen_classes,
                                        data.test_unseen_mask)
                val = m["i_zsl"]
                s = f"I-ZSL={val:.4f}"
            print(f"  Epoch {epoch:5d} | loss={loss.item():.4f} | {s}")

            if val > best_val:
                best_val, patience_ctr = val, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 50
                if patience_ctr >= args.patience:
                    break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        fp, fa = forward_fn()
    logits = get_logits(fp, fa).cpu()
    tm = data.test_seen_mask | data.test_unseen_mask

    zsl = compute_zsl_metrics(logits, data.y, data.unseen_classes,
                              data.test_unseen_mask)
    gzsl = compute_gzsl_metrics(logits, data.y, data.seen_classes,
                                data.unseen_classes, tm)

    print(f"\n{'='*60}")
    print(f"  {args.model} on {args.dataset}")
    print(f"  I-ZSL={zsl['i_zsl']:.4f}  S={gzsl['gzsl_s']:.4f}  "
          f"U={gzsl['gzsl_u']:.4f}  H={gzsl['harmonic_mean']:.4f}")
    print(f"{'='*60}")
    logger.log(model=args.model, dataset=args.dataset, seed=args.seed,
               **zsl, **gzsl)
    logger.save()
    return zsl, gzsl


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        cfg = yaml.safe_load(open(args.config))
        for sec in ["model", "training"]:
            for k, v in cfg.get(sec, {}).items():
                if hasattr(args, k):
                    setattr(args, k, v)
    train(args)