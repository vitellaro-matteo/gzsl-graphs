#!/usr/bin/env python3
"""
Upload all local datasets to your Hugging Face dataset repo.

This is a ONE-TIME setup script. After running it, anyone can
download via: python scripts/download_data.py

Only HF-hosted datasets are uploaded (cora, citeseer, C-M10-M, zerog).
The following are EXCLUDED because they have their own download pipelines:
  - ogbn-arxiv        (downloaded via OGB)
  - PubMed            (downloaded via PyG Planetoid)
  - WikiCS            (downloaded via PyG WikiCS)
  - Amazon/           (downloaded via PyG Amazon)
  - Coauthor/         (downloaded via PyG Coauthor)

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login   # authenticate with your HF token

Usage:
    python scripts/upload_to_hf.py --repo_id YOUR_USERNAME/gzsl-graphs-data

The script expects this local structure:
    data/
        cora/           cora.content, cora.cites
        citeseer/       citeseer.content, citeseer.cites
        C-M10-M/        feature.txt, graph.txt, group.txt, group-match.txt
        zerog/          cora.pt, citeseer.pt, pubmed.pt, arxiv.pt

It will upload everything under data/ to the HF repo, preserving structure.
"""

import argparse
import os
from pathlib import Path


# Directories to skip during upload — these datasets have their own
# download pipelines (OGB, PyG) and should not be uploaded to HF.
SKIP_DIRS = {
    "ogbn_arxiv", "ogbn-arxiv",    # OGB
    "PubMed", "pubmed",            # PyG Planetoid
    "WikiCS", "wikics",            # PyG WikiCS
    "Amazon",                       # PyG Amazon (Computers + Photo)
    "Coauthor",                     # PyG Coauthor (CS + Physics)
}


def _should_skip(dirpath: str) -> bool:
    """Check if any component of the path matches a skip directory."""
    parts = Path(dirpath).parts
    return any(p in SKIP_DIRS for p in parts)


def main():
    p = argparse.ArgumentParser(description="Upload datasets to HF Hub")
    p.add_argument("--repo_id", required=True,
                   help="HF dataset repo ID (e.g., mavitellaro/gzsl-graphs-data)")
    p.add_argument("--data_root", default="./data",
                   help="Local data directory to upload")
    p.add_argument("--private", action="store_true", default=False,
                   help="Create repo as private")
    a = p.parse_args()

    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(a.repo_id, repo_type="dataset", private=a.private)
        print(f"Created HF repo: {a.repo_id}")
    except Exception as e:
        if "already exists" in str(e).lower() or "409" in str(e):
            print(f"HF repo already exists: {a.repo_id}")
        else:
            raise

    data_root = Path(a.data_root)
    if not data_root.exists():
        print(f"ERROR: {data_root} does not exist.")
        raise SystemExit(1)

    # Collect all files to upload
    files = []
    skipped_dirs = set()
    for dirpath, dirnames, filenames in os.walk(data_root):
        rel_dir = os.path.relpath(dirpath, data_root)
        if _should_skip(rel_dir):
            skipped_dirs.add(rel_dir.split(os.sep)[0])
            continue
        for fname in filenames:
            local_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(local_path, data_root)
            files.append((local_path, rel_path))

    if skipped_dirs:
        print(f"\nSkipped directories (use their own download pipelines):")
        for d in sorted(skipped_dirs):
            print(f"  {d}/")

    if not files:
        print(f"\nNo files found to upload in {data_root}/")
        raise SystemExit(1)

    print(f"\nUploading {len(files)} files to hf://{a.repo_id}:")
    for local, remote in sorted(files):
        size_mb = os.path.getsize(local) / (1024 * 1024)
        print(f"  {remote} ({size_mb:.1f} MB)")

    print()
    for local, remote in files:
        print(f"  Uploading {remote}...")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=a.repo_id,
            repo_type="dataset",
        )

    print(f"\nDone! All files uploaded to: https://huggingface.co/datasets/{a.repo_id}")
    print(f"\nNow update HF_REPO_ID in scripts/download_data.py:")
    print(f'  HF_REPO_ID = "{a.repo_id}"')


if __name__ == "__main__":
    main()