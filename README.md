# Generalized Zero-Shot Learning on Graphs

> Investigating the GZSL performance of graph-based zero-shot node classification methods.

This repository accompanies the master thesis *"Generalized Zero-Shot Learning on Graphs"* (University of Luxembourg, January 2026) and its extension toward a research publication. It provides a clean, modular codebase for reproducing all experiments. The architecture is designed to scale: every model shares the same data loaders, evaluation pipeline, and experiment infrastructure.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Datasets](#datasets)
5. [Quick Start](#quick-start)
6. [Module Reference](#module-reference)
7. [Configuration](#configuration)
8. [Reproducing Results](#reproducing-results)
9. [Extending to New Models](#extending-to-new-models)
10. [Citation](#citation)

---

## Overview

Existing graph zero-shot learning methods (DGPN, DBiGCN, ZeroG) are evaluated only in the standard inductive ZSL setting, where test nodes come exclusively from unseen classes. This project extends evaluation to the **Generalized Zero-Shot Learning (GZSL)** setting, where test nodes come from *both* seen and unseen classes. The key finding is that existing methods exhibit a severe seen-class bias (H approximately 0%) under GZSL, which post-hoc calibration cannot rescue. Additionally, ICIS — a successful computer vision GZSL method — is adapted to graph data to test whether image-domain solutions transfer.

**Supported methods:** DGPN, DBiGCN, ZeroG, ICIS (adapted from computer vision)
**Supported datasets:** Cora, Citeseer, C-M10-M, ogbn-arXiv, PubMed (ZeroG only)
**Evaluation metrics:** I-ZSL accuracy, GZSL Seen (S), GZSL Unseen (U), Harmonic Mean (H)

The codebase follows [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) conventions: a library package (`gzsl_graphs/`) with subpackages for `data`, `models`, `transforms`, `evaluation`, and `utils`, plus thin executable scripts in `scripts/`.

---

## Project Structure

```
gzsl-graphs/
    README.md
    SETUP.md                               # HPC setup guide (micromamba, SLURM)
    environment.yaml                       # Conda/micromamba environment
    pyproject.toml                         # pip install -e . support
    .gitignore

    data/                                  # Dataset files (auto-downloaded)
        cora/                              #   cora.content, cora.cites
        citeseer/                          #   citeseer.content, citeseer.cites
        C-M10-M/                           #   feature.txt, graph.txt, group.txt
        ogbn_arxiv/                        #   (auto-created by OGB)
        zerog/                             #   Pre-processed .pt files for ZeroG

    configs/
        dgpn.yaml                          # DGPN hyperparameters
        dbigcn.yaml                        # DBiGCN hyperparameters
        zerog.yaml                         # ZeroG hyperparameters
        icis.yaml                          # ICIS hyperparameters

    gzsl_graphs/                           # Library package
        __init__.py
        data/                              # cf. torch_geometric.datasets
            __init__.py
            datasets.py                    # Unified loader for all 4 datasets
            splits.py                      # Fixed class splits (from DGPN paper)
            semantic.py                    # CSD encoding utilities
            zerog_subgraph.py              # k-hop subgraph extraction for ZeroG
        models/                            # cf. torch_geometric.nn.models
            __init__.py                    # Model registry + factory
            dgpn.py                        # Decomposed Graph Prototype Network
            dbigcn.py                      # Dual Bidirectional GCN
            zerog.py                       # ZeroG (LM + LoRA + prompt subgraphs)
            icis.py                        # ICIS joint autoencoder (adapted for graphs)
            icis_adapter.py                # Adapter: GraphZSLData -> ICIS format
        transforms/                        # cf. torch_geometric.transforms
            __init__.py
            decomposition.py               # Lazy random walk binomial decomposition
        evaluation/                        # Metrics + post-hoc analysis
            __init__.py
            metrics.py                     # I-ZSL, GZSL-S, GZSL-U, H
            calibration.py                 # Temperature scaling, gamma, grid search
        utils/                             # cf. torch_geometric.utils
            __init__.py
            logging.py                     # CSV/JSON experiment logging
            helpers.py                     # Seeds, device selection

    scripts/                               # Executable entry points
        download_data.py                   # Download all datasets (via HuggingFace Hub)
        upload_to_hf.py                    # One-time: upload data to HF Hub
        train.py                           # Train DGPN or DBiGCN
        train_zerog.py                     # Train ZeroG (standalone)
        train_icis.py                      # Train ICIS (standalone)
        evaluate.py                        # Standalone evaluation + calibration
        generate_csds.py                   # Generate ogbn-arXiv CSDs
        run_experiments.py                 # Batch runner from YAML config

    jobs/                                  # SLURM job scripts
        run_dgpn.sh
    tests/
        test_datasets.py
        test_metrics.py
    results/                               # Auto-generated outputs
```

### Design Principles

1. **One data loader for all models.** `GraphZSLDataset` returns identical `GraphZSLData` objects regardless of dataset or model. DGPN, DBiGCN, and ICIS all consume the same data via shared or adapted interfaces. ZeroG uses its own `.pt` files but shares the evaluation pipeline.

2. **Models are pure forward passes.** Each model implements `__init__` and `forward`. Loss functions are standalone. Models contain zero data loading or evaluation code.

3. **Transforms are reusable preprocessing.** The lazy random walk decomposition (DGPN) and adjacency construction (DBiGCN) are separate from models.

4. **Evaluation is fully decoupled.** Metrics and calibration operate on raw logits from any model. Calibration is post-hoc only and never affects training.

5. **Scripts are thin orchestrators.** Each model has its own training script: `train.py` (DGPN/DBiGCN), `train_zerog.py`, `train_icis.py`. All use the shared evaluation module.

---

## Installation

```bash
# Clone
git clone https://github.com/zhiqiangzhongddu/G-GZSL.git
cd gzsl_final/gzsl-graphs

# Create environment (micromamba or conda)
micromamba create -f environment.yaml -y
micromamba activate gzsl-graphs

# Install package (enables imports from anywhere)
pip install -e .

# Download all datasets
python scripts/download_data.py --data_root ./data

# Verify
python -c "import gzsl_graphs; print(gzsl_graphs.__version__)"
```

See `SETUP.md` for detailed HPC instructions (SLURM, CUDA version matching, offline setup).

---

## Datasets

All datasets can be downloaded with a single command via HuggingFace Hub:

```bash
python scripts/download_data.py                       # all datasets
python scripts/download_data.py --dataset cora        # just one
python scripts/download_data.py --dataset zerog       # ZeroG .pt files
python scripts/download_data.py --dataset ogbn-arxiv  # pre-download OGB
```

| Dataset    | Nodes   | Edges     | Classes | Seen / Unseen              | Used by |
|------------|---------|-----------|---------|----------------------------|---------|
| Cora       | 2,708   | 5,429     | 7       | 2 train / 2 val / 3 test  | All     |
| Citeseer   | 3,312   | 4,732     | 6       | 2 train / 2 val / 2 test  | All     |
| C-M10-M    | 4,464   | 5,804     | 6       | 2 train / 2 val / 2 test  | DGPN, DBiGCN, ICIS |
| ogbn-arXiv | 169,343 | 1,166,243 | 40      | 32 seen / 8 unseen         | All     |
| PubMed     | 19,717  | 44,338    | 3       | (ZeroG source only)        | ZeroG   |

### Class Semantic Descriptions (CSDs)

Each class has a rich text description encoded with SentenceBERT (`all-MiniLM-L6-v2`, 384-dim). All embeddings are L2-normalized. CSDs are generated automatically when loading a dataset for the first time.

---

## Quick Start

```bash
# DGPN on Cora (GZSL)
python scripts/train.py --model dgpn --dataset cora --data_root ./data --setting gzsl

# DBiGCN on Cora (GZSL)
python scripts/train.py --model dbigcn --dataset cora --data_root ./data --setting gzsl

# ZeroG: cross-dataset transfer (train on citation, test on Cora/Citeseer/Pubmed)
python scripts/train_zerog.py \
    --source Cora Citeseer Pubmed Arxiv \
    --target Cora Citeseer Pubmed \
    --data_dir ./data/zerog --epochs 50

# ICIS adapted to graphs (Chapter 5)
python scripts/train_icis.py --dataset cora --data_root ./data --epochs 1000

# Run all DGPN experiments
python scripts/run_experiments.py --config configs/dgpn.yaml

# Run all DBiGCN experiments
python scripts/run_experiments.py --config configs/dbigcn.yaml
```

---

## Module Reference

### `gzsl_graphs.data.datasets.GraphZSLDataset`

Unified loader returning `GraphZSLData` with: `x`, `edge_index`, `y`, `class_semantics`, `seen_classes`, `unseen_classes`, `val_classes`, masks, `target_weights`, `class_names`. Backward-compatible properties: `.seenclasses`, `.unseenclasses`, `.attribute`, `.features`, `.labels`.

### `gzsl_graphs.models.dgpn.DGPN`

`DGPN(feature_list, csd_matrix) -> (preds, local_preds)`. Single shared encoder, sum pooling. Loss via `compute_dgpn_loss()`. Requires lazy random walk decomposition as preprocessing.

### `gzsl_graphs.models.dbigcn.DBiGCN`

`DBiGCN(X, CSD, S_V, S_A) -> (Y_X, Y_A)`. Dual branch: BiGCN_X (node perspective, 2-layer MLP) and BiGCN_A (class perspective, 1-layer MLP). Loss via `compute_dbigcn_loss()`. Requires normalized node and class adjacency matrices. Prediction uses Y_X only (Eq. 10 in paper).

### `gzsl_graphs.models.zerog.ZeroG`

SentenceBERT with LoRA fine-tuning. Training: k-hop subgraphs with virtual prompt node, R rounds of neighbor aggregation, cross-entropy with label embeddings. Inference: `encode_graph(data, name) -> [N, C] logits`. Uses raw text per node (not pre-computed features). Separate training script: `scripts/train_zerog.py`.

### `gzsl_graphs.models.icis.JointAutoencoder`

ICIS core: two autoencoders (semantic attributes and classifier weights) sharing a latent space. Training: 4 reconstruction losses (within-space + cross-space). Inference: `predict(attributes) -> weights` maps unseen class descriptions to classifier weight vectors. Adapted from computer vision (ResNet -> linear classifier on graph features, visual attributes -> SentenceBERT CSDs). Separate training script: `scripts/train_icis.py`.

### `gzsl_graphs.transforms.decomposition`

Binomial decomposition for DGPN (Eq. 13): `lazy_random_walk_decompose(edge_index, x, K, beta)`.

### `gzsl_graphs.evaluation.metrics`

`compute_zsl_metrics(logits, labels, unseen_classes, test_mask)` and `compute_gzsl_metrics(logits, labels, seen, unseen, test_mask)`.

### `gzsl_graphs.evaluation.calibration`

Post-hoc analysis only (never called during training). `calibration_grid_search(logits, ...)` searches over (T, gamma) for best H. `plot_calibration_heatmap(result)` produces thesis-style heatmaps.

---

## Configuration

```yaml
# configs/dgpn.yaml
data_root: ./data
model:
  name: dgpn
  K: 3
  beta: 0.7
  alpha: 1.0
  dropout: 0.5
training:
  epochs: 10000
  lr: 0.001
datasets: [cora, citeseer, c-m10-m, ogbn-arxiv]
```

```yaml
# configs/dbigcn.yaml
data_root: ./data
model:
  name: dbigcn
  hidden_dim: 512
  dropout: 0
  n_neighbors: 2
training:
  epochs: 10000
  lr: 0.0001
  wd: 0.0001
  alpha: 1.0
  loss_beta: 1.0
datasets: [cora, citeseer, c-m10-m, ogbn-arxiv]
```

```yaml
# configs/icis.yaml
data_root: ./data
model:
  name: icis
  embed_dim: 1000
  num_layers: 3
  wn_factor: 10.0
training:
  epochs: 1000
  lr: 0.0001
  batch_size: 16
datasets: [cora, citeseer, c-m10-m, ogbn-arxiv]
```

---

## Reproducing Results

### DGPN (Table 4.3-4.5)

```bash
python scripts/run_experiments.py --config configs/dgpn.yaml
```

| Dataset    | I-ZSL  | GZSL-S | GZSL-U | H      |
|------------|--------|--------|--------|--------|
| Cora       | 33.96% | 97.89% | 0.40%  | 0.80%  |
| Citeseer   | 74.95% | 82.74% | 0.55%  | 1.09%  |
| C-M10-M    | 57.52% | 93.13% | 0.27%  | 0.55%  |
| ogbn-arXiv | 18.06% | 62.69% | 0.03%  | 0.06%  |

### DBiGCN (Table 4.7)

```bash
python scripts/run_experiments.py --config configs/dbigcn.yaml
```

Expected: H = 0.00% for Cora/Citeseer/ogbn-arXiv, H > 0 for C-M10-M.

### ZeroG (Section 4.7.2)

```bash
python scripts/train_zerog.py \
    --source Cora Citeseer Pubmed Arxiv \
    --target Cora Citeseer Pubmed \
    --data_dir ./data/zerog --epochs 50
```

Note: ZeroG uses a fundamentally different paradigm (cross-dataset transfer with LM fine-tuning). PubMed generates 93% of training subgraphs due to high connectivity, causing data imbalance that is documented as a finding rather than corrected.

### ICIS (Chapter 5)

```bash
# Small citation networks (thesis reports H~38% on Cora/Citeseer with calibration)
python scripts/train_icis.py --dataset cora --data_root ./data --epochs 1000
python scripts/train_icis.py --dataset citeseer --data_root ./data --epochs 1000

# Large-scale (thesis reports near-zero H, demonstrating scale-dependent fragility)
python scripts/train_icis.py --dataset ogbn-arxiv --data_root ./data --epochs 1000
```

Note: ICIS is adapted from computer vision (Christensen et al., ICCV 2023). The adaptation replaces ResNet features with raw graph node features and visual attributes with SentenceBERT CSDs. Performance degrades at scale due to the semantic-to-structure mapping gap (see thesis Section 5.2).

---

## Extending to New Models

To add a new model:

1. Create `gzsl_graphs/models/newmodel.py` with model architecture
2. Add `configs/newmodel.yaml`
3. Register in `gzsl_graphs/models/__init__.py`
4. Create `scripts/train_newmodel.py` (standalone) or add to `scripts/train.py`
5. Run: `python scripts/train_newmodel.py --dataset cora`

The data loader, metrics, calibration, and logging require **zero changes**. For models with incompatible data formats, use the adapter pattern (see `icis_adapter.py`).

---

## Citation

```bibtex
@mastersthesis{vitellaro2026gzsl,
  title   = {Generalized Zero-Shot Learning on Graphs},
  author  = {Vitellaro, Matteo},
  school  = {University of Luxembourg},
  year    = {2026},
  month   = {January}
}
```

### References

- **DGPN:** Wang et al., KDD 2021. [github.com/zhengwang100/dgpn](https://github.com/zhengwang100/dgpn)
- **DBiGCN:** Yue et al., KDD 2022. [github.com/warmerspring/DBiGCN](https://github.com/warmerspring/DBiGCN)
- **ZeroG:** Li et al., KDD 2024. [github.com/NineAbyss/ZeroG](https://github.com/NineAbyss/ZeroG)
- **ICIS:** Christensen et al., ICCV 2023. [github.com/ExplainableML/ImageFreeZSL](https://github.com/ExplainableML/ImageFreeZSL)