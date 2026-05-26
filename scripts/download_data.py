#!/usr/bin/env python3
"""
Download all datasets for graph zero-shot learning.

Datasets fall into two categories:

  HF-hosted (custom formats, uploaded via upload_to_hf.py):
    cora/           cora.content, cora.cites
    citeseer/       citeseer.content, citeseer.cites
    C-M10-M/        feature.txt, graph.txt, group.txt, group-match.txt
    zerog/          cora.pt, citeseer.pt, pubmed.pt, arxiv.pt

  Auto-downloaded (via PyTorch Geometric or OGB):
    ogbn-arxiv      downloaded via OGB
    pubmed          downloaded via PyG Planetoid
    wikics          downloaded via PyG WikiCS
    amazon-computers downloaded via PyG Amazon
    amazon-photo     downloaded via PyG Amazon
    coauthor-cs      downloaded via PyG Coauthor
    coauthor-physics downloaded via PyG Coauthor

Usage:
    python scripts/download_data.py                            # all datasets
    python scripts/download_data.py --dataset cora             # just one
    python scripts/download_data.py --dataset zerog            # ZeroG .pt files
    python scripts/download_data.py --dataset amazon-computers # PyG auto-download
    python scripts/download_data.py --repo_id user/repo        # override HF repo

Setup:
    pip install huggingface_hub torch_geometric ogb
    # If HF repo is private: huggingface-cli login
"""

import argparse, os
from pathlib import Path

HF_REPO_ID = "vitellaro-matteo/gzsl-graphs-data"


# ============================================================
# HF-hosted datasets
# ============================================================

def _hf_download(repo_id, subfolder, local_dir):
    """Download a subfolder from a HF dataset repo."""
    from huggingface_hub import snapshot_download
    print(f"  Downloading {subfolder}/ from hf://{repo_id} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=[f"{subfolder}/**"],
    )


def download_cora(root, repo_id):
    if (Path(root) / "cora" / "cora.content").exists():
        print("  Cora already present, skipping."); return
    _hf_download(repo_id, "cora", root)
    print(f"  Cora ready at {root}/cora/")


def download_citeseer(root, repo_id):
    if (Path(root) / "citeseer" / "citeseer.content").exists():
        print("  Citeseer already present, skipping."); return
    _hf_download(repo_id, "citeseer", root)
    print(f"  Citeseer ready at {root}/citeseer/")


def download_cm10m(root, repo_id):
    if (Path(root) / "C-M10-M" / "feature.txt").exists():
        print("  C-M10-M already present, skipping."); return
    _hf_download(repo_id, "C-M10-M", root)
    print(f"  C-M10-M ready at {root}/C-M10-M/")


def download_zerog(root, repo_id):
    """Download ZeroG pre-processed .pt files."""
    d = Path(root) / "zerog"
    existing = list(d.glob("*.pt")) if d.exists() else []
    if len(existing) >= 3:
        print(f"  ZeroG data already present ({len(existing)} .pt files), skipping."); return
    d.mkdir(parents=True, exist_ok=True)
    _hf_download(repo_id, "zerog", root)
    pt_files = list(d.glob("*.pt"))
    print(f"  ZeroG ready at {d}/ ({len(pt_files)} .pt files)")


# ============================================================
# OGB-hosted dataset
# ============================================================

def download_ogbn_arxiv(root, repo_id):
    """ogbn-arXiv via OGB (has its own download pipeline)."""
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError:
        print("  OGB not installed. Run: pip install ogb")
        print("  ogbn-arXiv will auto-download on first training run."); return
    d = Path(root); d.mkdir(parents=True, exist_ok=True)
    print("  Downloading ogbn-arXiv via OGB...")
    os.environ["OGB_DATA_HOME"] = str(d)
    NodePropPredDataset(name="ogbn-arxiv", root=str(d))
    print(f"  ogbn-arXiv ready at {d}")


# ============================================================
# PyG-hosted datasets (auto-download from PyG/GitHub)
# ============================================================

def _check_pyg_installed():
    """Check that torch_geometric is available."""
    try:
        import torch_geometric  # noqa: F401
        return True
    except ImportError:
        print("  PyTorch Geometric not installed. Run: pip install torch_geometric")
        print("  Dataset will auto-download on first training run if PyG is available.")
        return False


def download_pubmed(root, repo_id):
    """PubMed citation network via PyG Planetoid."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "PubMed" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  PubMed already present, skipping."); return
    from torch_geometric.datasets import Planetoid
    print("  Downloading PubMed via PyG Planetoid...")
    Planetoid(root=root, name="PubMed")
    print(f"  PubMed ready at {root}/PubMed/")


def download_wikics(root, repo_id):
    """WikiCS dataset via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "WikiCS" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  WikiCS already present, skipping."); return
    from torch_geometric.datasets import WikiCS
    print("  Downloading WikiCS via PyG...")
    WikiCS(root=os.path.join(root, "WikiCS"), is_undirected=True)
    print(f"  WikiCS ready at {root}/WikiCS/")


def download_amazon_computers(root, repo_id):
    """Amazon Computers co-purchase graph via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "Amazon" / "Computers" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  Amazon Computers already present, skipping."); return
    from torch_geometric.datasets import Amazon
    print("  Downloading Amazon Computers via PyG...")
    Amazon(root=os.path.join(root, "Amazon"), name="Computers")
    print(f"  Amazon Computers ready at {root}/Amazon/Computers/")


def download_amazon_photo(root, repo_id):
    """Amazon Photo co-purchase graph via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "Amazon" / "Photo" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  Amazon Photo already present, skipping."); return
    from torch_geometric.datasets import Amazon
    print("  Downloading Amazon Photo via PyG...")
    Amazon(root=os.path.join(root, "Amazon"), name="Photo")
    print(f"  Amazon Photo ready at {root}/Amazon/Photo/")


def download_coauthor_cs(root, repo_id):
    """Coauthor CS co-authorship graph via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "Coauthor" / "CS" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  Coauthor CS already present, skipping."); return
    from torch_geometric.datasets import Coauthor
    print("  Downloading Coauthor CS via PyG...")
    Coauthor(root=os.path.join(root, "Coauthor"), name="CS")
    print(f"  Coauthor CS ready at {root}/Coauthor/CS/")


def download_coauthor_physics(root, repo_id):
    """Coauthor Physics co-authorship graph via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "Coauthor" / "Physics" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  Coauthor Physics already present, skipping."); return
    from torch_geometric.datasets import Coauthor
    print("  Downloading Coauthor Physics via PyG...")
    Coauthor(root=os.path.join(root, "Coauthor"), name="Physics")
    print(f"  Coauthor Physics ready at {root}/Coauthor/Physics/")


# ============================================================
# Registry
# ============================================================

ALL = {
    # HF-hosted
    "cora": download_cora,
    "citeseer": download_citeseer,
    "c-m10-m": download_cm10m,
    "zerog": download_zerog,
    # OGB-hosted
    "ogbn-arxiv": download_ogbn_arxiv,
    # PyG-hosted
    "pubmed": download_pubmed,
    "wikics": download_wikics,
    "amazon-computers": download_amazon_computers,
    "amazon-photo": download_amazon_photo,
    "coauthor-cs": download_coauthor_cs,
    "coauthor-physics": download_coauthor_physics,
}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download datasets for graph ZSL")
    p.add_argument("--dataset", default=None, choices=list(ALL.keys()))
    p.add_argument("--data_root", default="./data")
    p.add_argument("--repo_id", default=None, help="Override HF repo ID")
    a = p.parse_args()

    repo_id = a.repo_id or HF_REPO_ID
    if "YOUR_HF_USERNAME" in repo_id:
        print("ERROR: Set HF_REPO_ID in this script or pass --repo_id.")
        print("See scripts/upload_to_hf.py for setup instructions.")
        raise SystemExit(1)

    print(f"HF Repo: {repo_id}")
    print(f"Data root: {os.path.abspath(a.data_root)}")
    for ds in ([a.dataset] if a.dataset else list(ALL.keys())):
        print(f"\n{'='*50}\n  {ds}\n{'='*50}")
        ALL[ds](a.data_root, repo_id)
    print(f"\nDone. Datasets in {os.path.abspath(a.data_root)}/")