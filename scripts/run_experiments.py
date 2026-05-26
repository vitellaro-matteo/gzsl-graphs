#!/usr/bin/env python3
"""
Unified experiment runner for all graph GZSL models and datasets.

Dispatches to the appropriate training script for each model, runs all
dataset x seed combinations, collects results into a single JSON, and
optionally generates a LaTeX results table.

Models:
    dgpn, dbigcn       -> scripts/train.py
    icis               -> scripts/train_icis.py
    baseline           -> scripts/train_baseline.py
    zerog              -> scripts/train_zerog.py  (separate workflow)

Usage:
    # Run everything from a YAML config
    python scripts/run_experiments.py --config configs/full_benchmark.yaml

    # Override specific settings
    python scripts/run_experiments.py --config configs/full_benchmark.yaml \
        --models dgpn dbigcn --datasets cora citeseer --seeds 0 1 2

    # Single model, single dataset (quick test)
    python scripts/run_experiments.py --models dgpn --datasets cora --seeds 0

    # Aggregate existing results only (no training)
    python scripts/run_experiments.py --aggregate_only --output_dir results/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


# ============================================================
# Dataset and model definitions
# ============================================================

ALL_DATASETS = [
    "cora", "citeseer", "c-m10-m", "ogbn-arxiv",
    "pubmed", "wikics", "amazon-computers", "amazon-photo",
    "coauthor-cs", "coauthor-physics",
]

# Models that use scripts/train.py (shared training loop)
TRAIN_PY_MODELS = ["dgpn", "dbigcn"]

# Models with their own training scripts
STANDALONE_MODELS = {
    "icis": "scripts/train_icis.py",
    "baseline": "scripts/train_baseline.py",
}

# ZeroG is cross-dataset, handled separately
ZEROG_SCRIPT = "scripts/train_zerog.py"

ALL_MODELS = TRAIN_PY_MODELS + list(STANDALONE_MODELS.keys()) + ["zerog"]

DEFAULT_SEEDS = [0, 1, 2, 3, 4]


# ============================================================
# Command builders for each model type
# ============================================================

def build_train_py_cmd(model, dataset, seed, cfg, data_root):
    """Build command for DGPN/DBiGCN via scripts/train.py."""
    cmd = [
        sys.executable, "scripts/train.py",
        "--model", model,
        "--dataset", dataset,
        "--setting", "gzsl",
        "--seed", str(seed),
        "--data_root", data_root,
        "--output_dir", f"results/{model}/",
    ]

    # Model-specific params from config
    model_cfg = cfg.get("models", {}).get(model, {})
    train_cfg = model_cfg.get("training", {})
    arch_cfg = model_cfg.get("architecture", {})

    param_map = {
        "epochs": "--epochs", "lr": "--lr", "wd": "--wd",
        "alpha": "--alpha", "loss_beta": "--loss_beta",
        "K": "--K", "beta": "--beta", "dropout": "--dropout",
        "hidden_dim": "--hidden_dim", "n_neighbors": "--n_neighbors",
        "patience": "--patience",
    }
    for src_key, flag in param_map.items():
        for d in [train_cfg, arch_cfg]:
            if src_key in d:
                cmd += [flag, str(d[src_key])]
                break

    return cmd


def build_icis_cmd(dataset, seed, cfg, data_root):
    """Build command for ICIS via scripts/train_icis.py."""
    cmd = [
        sys.executable, "scripts/train_icis.py",
        "--dataset", dataset,
        "--seed", str(seed),
        "--data_root", data_root,
        "--output_dir", f"results/icis/",
    ]

    icis_cfg = cfg.get("models", {}).get("icis", {})
    for key in ["classifier_epochs", "classifier_lr", "epochs", "lr",
                "batch_size", "embed_dim", "num_layers", "wn_factor",
                "patience"]:
        if key in icis_cfg:
            cmd += [f"--{key}", str(icis_cfg[key])]
    if icis_cfg.get("cos_sim_loss", False):
        cmd += ["--cos_sim_loss"]

    return cmd


def build_baseline_cmd(dataset, seed, cfg, data_root):
    """Build command for transductive baseline via scripts/train_baseline.py."""
    cmd = [
        sys.executable, "scripts/train_baseline.py",
        "--dataset", dataset,
        "--seeds", str(seed),
        "--data_root", data_root,
        "--output_dir", f"results/baseline/",
    ]

    bl_cfg = cfg.get("models", {}).get("baseline", {})
    for key in ["enhanced_dim", "epochs_phase1", "lr_phase1",
                "joint_dim", "epochs_phase2", "lr_phase2",
                "lambda_align", "temperature", "max_iterations"]:
        if key in bl_cfg:
            cmd += [f"--{key}", str(bl_cfg[key])]

    return cmd


def build_zerog_cmd(source, target, seed, cfg, data_root):
    """Build command for ZeroG via scripts/train_zerog.py."""
    cmd = [
        sys.executable, "scripts/train_zerog.py",
        "--source"] + source + [
        "--target"] + target + [
        "--seed", str(seed),
        "--data_dir", os.path.join(data_root, "zerog"),
        "--output_dir", f"results/zerog/",
    ]

    zg_cfg = cfg.get("models", {}).get("zerog", {})
    for key in ["epochs", "lr", "k", "R", "batch_size", "grad_accum"]:
        if key in zg_cfg:
            cmd += [f"--{key}", str(zg_cfg[key])]

    return cmd


# ============================================================
# Execution
# ============================================================

def run_cmd(cmd, dry_run=False):
    """Run a subprocess command, return success bool."""
    cmd_str = " ".join(cmd)
    if dry_run:
        print(f"  [DRY RUN] {cmd_str}")
        return True

    print(f"  Running: {cmd_str}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}) after {elapsed:.1f}s")
        return False
    else:
        print(f"  OK ({elapsed:.1f}s)")
        return True


def run_standard_models(models, datasets, seeds, cfg, data_root, dry_run=False):
    """Run DGPN, DBiGCN, ICIS, baseline across datasets x seeds."""
    results_log = []

    for model in models:
        if model == "zerog":
            continue  # handled separately

        for dataset in datasets:
            for seed in seeds:
                print(f"\n{'='*60}")
                print(f"  {model} | {dataset} | seed={seed}")
                print(f"{'='*60}")

                if model in TRAIN_PY_MODELS:
                    cmd = build_train_py_cmd(model, dataset, seed, cfg, data_root)
                elif model == "icis":
                    cmd = build_icis_cmd(dataset, seed, cfg, data_root)
                elif model == "baseline":
                    cmd = build_baseline_cmd(dataset, seed, cfg, data_root)
                else:
                    print(f"  Unknown model: {model}, skipping")
                    continue

                ok = run_cmd(cmd, dry_run)
                results_log.append({
                    "model": model, "dataset": dataset,
                    "seed": seed, "success": ok,
                })

    return results_log


def run_zerog(zerog_cfg, seeds, data_root, dry_run=False):
    """Run ZeroG experiments (cross-dataset transfer)."""
    results_log = []
    zg = zerog_cfg or {}
    source = zg.get("source", ["Cora", "Citeseer", "Pubmed", "Arxiv"])
    targets = zg.get("target", ["Cora", "Citeseer", "Pubmed"])

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  zerog | source={source} | target={targets} | seed={seed}")
        print(f"{'='*60}")

        # Build a minimal cfg wrapper for build_zerog_cmd
        cfg = {"models": {"zerog": zg}}
        cmd = build_zerog_cmd(source, targets, seed, cfg, data_root)
        ok = run_cmd(cmd, dry_run)
        results_log.append({
            "model": "zerog", "source": source,
            "target": targets, "seed": seed, "success": ok,
        })

    return results_log


# ============================================================
# Aggregation
# ============================================================

def aggregate_results(output_dir):
    """Scan results/ directories and aggregate all JSON logs.

    Expects each training script to save results via ExperimentLogger,
    producing JSON files with keys like:
        model, dataset, seed, i_zsl, gzsl_s, gzsl_u, harmonic_mean
    """
    results = defaultdict(lambda: defaultdict(list))
    results_dir = Path(output_dir)

    if not results_dir.exists():
        print(f"No results directory found at {output_dir}")
        return {}

    for json_path in results_dir.rglob("*.json"):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # Handle both single-result and list-of-results formats
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model = entry.get("model", "unknown")
            dataset = entry.get("dataset", "unknown")
            key = (model, dataset)

            metrics = {}
            for m in ["i_zsl", "gzsl_s", "gzsl_u", "harmonic_mean"]:
                if m in entry:
                    metrics[m] = entry[m]
            if metrics:
                results[key]["metrics"].append(metrics)

    return results


def print_results_table(results):
    """Print a formatted results table to stdout."""
    if not results:
        print("No results to aggregate.")
        return

    print(f"\n{'='*100}")
    print(f"  AGGREGATED RESULTS (mean ± std over seeds)")
    print(f"{'='*100}")
    header = f"{'Model':<12} {'Dataset':<20} {'I-ZSL':>12} {'GZSL-S':>12} {'GZSL-U':>12} {'H':>12}"
    print(header)
    print("-" * 100)

    for (model, dataset), data in sorted(results.items()):
        metrics_list = data.get("metrics", [])
        if not metrics_list:
            continue

        row = f"{model:<12} {dataset:<20}"
        for key in ["i_zsl", "gzsl_s", "gzsl_u", "harmonic_mean"]:
            vals = [m[key] for m in metrics_list if key in m]
            if vals:
                mean = np.mean(vals) * 100
                std = np.std(vals) * 100
                row += f" {mean:5.1f}±{std:4.1f}  "
            else:
                row += f" {'N/A':>11} "
        print(row)

    print(f"{'='*100}")


def save_latex_table(results, output_path):
    """Generate a LaTeX table from aggregated results."""
    if not results:
        return

    # Group by dataset for the table layout the professor wants
    models_seen = sorted(set(m for m, _ in results.keys()))
    datasets_seen = sorted(set(d for _, d in results.keys()))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{GZSL results on graph benchmarks (mean$\pm$std over 5 seeds).}",
        r"\label{tab:gzsl_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "cccc" * len(models_seen) + "}",
        r"\toprule",
    ]

    # Header row 1: model names spanning 4 columns each
    header1 = "Dataset"
    for m in models_seen:
        header1 += f" & \\multicolumn{{4}}{{c}}{{{m.upper()}}}"
    header1 += r" \\"
    lines.append(header1)

    # Header row 2: metric names
    header2 = ""
    for _ in models_seen:
        header2 += " & I-ZSL & S & U & H"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    # Data rows
    for ds in datasets_seen:
        row = ds.replace("-", "\\text{-}")
        for m in models_seen:
            key = (m, ds)
            if key in results and results[key].get("metrics"):
                metrics_list = results[key]["metrics"]
                for metric_key in ["i_zsl", "gzsl_s", "gzsl_u", "harmonic_mean"]:
                    vals = [x[metric_key] for x in metrics_list if metric_key in x]
                    if vals:
                        mean = np.mean(vals) * 100
                        std = np.std(vals) * 100
                        row += f" & {mean:.1f}$\\pm${std:.1f}"
                    else:
                        row += " & --"
            else:
                row += " & -- & -- & -- & --"
        row += r" \\"
        lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nLaTeX table saved to {output_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified experiment runner for all graph GZSL models")
    p.add_argument("--config", default=None,
                   help="YAML config file (see configs/full_benchmark.yaml)")
    p.add_argument("--models", nargs="+", default=None,
                   choices=ALL_MODELS,
                   help="Override: which models to run")
    p.add_argument("--datasets", nargs="+", default=None,
                   choices=ALL_DATASETS,
                   help="Override: which datasets to run")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Override: which seeds to use")
    p.add_argument("--data_root", default=None)
    p.add_argument("--output_dir", default="results/")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--aggregate_only", action="store_true",
                   help="Only aggregate existing results, don't train")
    p.add_argument("--latex", default=None,
                   help="Path to save LaTeX table (e.g., results/table.tex)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # Resolve parameters (CLI overrides config)
    models = args.models or cfg.get("models_to_run", TRAIN_PY_MODELS)
    datasets = args.datasets or cfg.get("datasets", ["cora", "citeseer"])
    seeds = args.seeds or cfg.get("seeds", DEFAULT_SEEDS)
    data_root = args.data_root or cfg.get("data_root", "./data")

    print(f"Models:   {models}")
    print(f"Datasets: {datasets}")
    print(f"Seeds:    {seeds}")
    print(f"Data:     {data_root}")

    if not args.aggregate_only:
        # Run standard models (DGPN, DBiGCN, ICIS, baseline)
        std_models = [m for m in models if m != "zerog"]
        if std_models:
            run_standard_models(std_models, datasets, seeds, cfg,
                                data_root, args.dry_run)

        # Run ZeroG (cross-dataset, separate workflow)
        if "zerog" in models:
            zerog_cfg = cfg.get("models", {}).get("zerog", {})
            run_zerog(zerog_cfg, seeds, data_root, args.dry_run)

    # Aggregate and display
    results = aggregate_results(args.output_dir)
    print_results_table(results)

    if args.latex:
        save_latex_table(results, args.latex)

    # Save raw aggregated JSON
    agg_path = os.path.join(args.output_dir, "aggregated_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    serializable = {}
    for (model, dataset), data in results.items():
        key = f"{model}__{dataset}"
        metrics = data.get("metrics", [])
        summary = {}
        for m in ["i_zsl", "gzsl_s", "gzsl_u", "harmonic_mean"]:
            vals = [x[m] for x in metrics if m in x]
            if vals:
                summary[m] = {"mean": float(np.mean(vals)),
                              "std": float(np.std(vals)),
                              "values": vals}
        serializable[key] = summary

    with open(agg_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Aggregated results saved to {agg_path}")


if __name__ == "__main__":
    main()