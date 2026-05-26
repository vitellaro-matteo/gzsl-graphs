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
# Extended datasets (added for large-scale benchmarking)
# ============================================================

def download_cora_full(root, repo_id):
    """CoraFull 70-class citation graph via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "CoraFull" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  CoraFull already present, skipping."); return
    from torch_geometric.datasets import CoraFull
    print("  Downloading CoraFull via PyG...")
    CoraFull(root=os.path.join(root, "CoraFull"))
    print(f"  CoraFull ready at {root}/CoraFull/")


def download_ogbn_products(root, repo_id):
    """ogbn-products 47-class Amazon co-purchase graph via OGB."""
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError:
        print("  OGB not installed. Run: pip install ogb"); return
    d = Path(root); d.mkdir(parents=True, exist_ok=True)
    # ogbn-products is ~8 GB; skip if already present
    existing = list((d / "ogbn_products").rglob("*.npz")) + list((d / "ogbn_products").rglob("*.csv.gz"))
    if existing:
        print("  ogbn-products already present, skipping."); return
    print("  Downloading ogbn-products via OGB (~8 GB, may take a while)...")
    os.environ["OGB_DATA_HOME"] = str(d)
    NodePropPredDataset(name="ogbn-products", root=str(d))
    print(f"  ogbn-products ready at {d}")


def download_reddit(root, repo_id):
    """Reddit2 41-class subreddit graph via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "Reddit2" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  Reddit2 already present, skipping."); return
    from torch_geometric.datasets import Reddit2
    print("  Downloading Reddit2 via PyG (~1 GB)...")
    Reddit2(root=os.path.join(root, "Reddit2"))
    print(f"  Reddit2 ready at {root}/Reddit2/")


def download_roman_empire(root, repo_id):
    """Roman-empire heterophilous graph via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "HeterophilousGraphDataset" / "Roman-empire" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  Roman-empire already present, skipping."); return
    from torch_geometric.datasets import HeterophilousGraphDataset
    print("  Downloading Roman-empire via PyG...")
    HeterophilousGraphDataset(root=os.path.join(root, "HeterophilousGraphDataset"), name="Roman-empire")
    print(f"  Roman-empire ready at {root}/HeterophilousGraphDataset/")


def download_flickr(root, repo_id):
    """Flickr image-graph via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "Flickr" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  Flickr already present, skipping."); return
    from torch_geometric.datasets import Flickr
    print("  Downloading Flickr via PyG (~300 MB)...")
    Flickr(root=os.path.join(root, "Flickr"))
    print(f"  Flickr ready at {root}/Flickr/")


def download_lastfm_asia(root, repo_id):
    """LastFMAsia social graph via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "LastFMAsia" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  LastFMAsia already present, skipping."); return
    from torch_geometric.datasets import LastFMAsia
    print("  Downloading LastFMAsia via PyG...")
    LastFMAsia(root=os.path.join(root, "LastFMAsia"))
    print(f"  LastFMAsia ready at {root}/LastFMAsia/")


def download_actor(root, repo_id):
    """Actor film co-occurrence graph via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "Actor" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  Actor already present, skipping."); return
    from torch_geometric.datasets import Actor
    print("  Downloading Actor via PyG...")
    Actor(root=os.path.join(root, "Actor"))
    print(f"  Actor ready at {root}/Actor/")


def download_chameleon(root, repo_id):
    """Chameleon Wikipedia network via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "WikipediaNetwork" / "chameleon" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  Chameleon already present, skipping."); return
    from torch_geometric.datasets import WikipediaNetwork
    print("  Downloading Chameleon via PyG...")
    WikipediaNetwork(root=os.path.join(root, "WikipediaNetwork"), name="chameleon",
                     geom_gcn_preprocess=True)
    print(f"  Chameleon ready at {root}/WikipediaNetwork/")


def download_squirrel(root, repo_id):
    """Squirrel Wikipedia network via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "WikipediaNetwork" / "squirrel" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  Squirrel already present, skipping."); return
    from torch_geometric.datasets import WikipediaNetwork
    print("  Downloading Squirrel via PyG...")
    WikipediaNetwork(root=os.path.join(root, "WikipediaNetwork"), name="squirrel",
                     geom_gcn_preprocess=True)
    print(f"  Squirrel ready at {root}/WikipediaNetwork/")


def download_amazon_ratings(root, repo_id):
    """Amazon-ratings heterophilous graph via PyG."""
    if not _check_pyg_installed():
        return
    target = Path(root) / "HeterophilousGraphDataset" / "Amazon-ratings" / "processed"
    if target.exists() and list(target.glob("*.pt")):
        print("  Amazon-ratings already present, skipping."); return
    from torch_geometric.datasets import HeterophilousGraphDataset
    print("  Downloading Amazon-ratings via PyG...")
    HeterophilousGraphDataset(root=os.path.join(root, "HeterophilousGraphDataset"),
                              name="Amazon-ratings")
    print(f"  Amazon-ratings ready at {root}/HeterophilousGraphDataset/")


# ============================================================
# Registry
# ============================================================

ALL = {
    # HF-hosted (custom formats)
    "cora": download_cora,
    "citeseer": download_citeseer,
    "c-m10-m": download_cm10m,
    "zerog": download_zerog,
    # OGB-hosted
    "ogbn-arxiv": download_ogbn_arxiv,
    "ogbn-products": download_ogbn_products,
    # PyG standard
    "pubmed": download_pubmed,
    "wikics": download_wikics,
    "amazon-computers": download_amazon_computers,
    "amazon-photo": download_amazon_photo,
    "coauthor-cs": download_coauthor_cs,
    "coauthor-physics": download_coauthor_physics,
    # Extended benchmarks
    "cora-full": download_cora_full,
    "reddit": download_reddit,
    "roman-empire": download_roman_empire,
    "flickr": download_flickr,
    "lastfm-asia": download_lastfm_asia,
    "actor": download_actor,
    "chameleon": download_chameleon,
    "squirrel": download_squirrel,
    "amazon-ratings": download_amazon_ratings,
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