"""
Class Semantic Description (CSD) utilities.

The main CSD logic is handled inside GraphZSLDataset.load(). This module
exposes description dictionaries and a standalone encode function for
scripts like generate_csds.py.
"""

import torch
from typing import List

from .datasets import (
    CORA_DESCRIPTIONS, CITESEER_DESCRIPTIONS,
    CM10M_DESCRIPTIONS, ARXIV_DESCRIPTIONS, ARXIV_CATEGORY_NAMES,
    PUBMED_DESCRIPTIONS, WIKICS_DESCRIPTIONS,
    AMAZON_COMPUTERS_DESCRIPTIONS, AMAZON_PHOTO_DESCRIPTIONS,
    COAUTHOR_CS_DESCRIPTIONS, COAUTHOR_PHYSICS_DESCRIPTIONS,
)


def encode_descriptions(descriptions: List[str],
                        model_name: str = "all-MiniLM-L6-v2") -> torch.Tensor:
    """Encode text descriptions with SentenceBERT, L2-normalize.

    Returns:
        Tensor [len(descriptions), embedding_dim], L2-normalized.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(descriptions, convert_to_tensor=True)
    return emb / emb.norm(dim=1, keepdim=True)


def load_class_semantics(dataset_name: str, csd_type: str = "text",
                         model_name: str = "all-MiniLM-L6-v2") -> torch.Tensor:
    """Standalone CSD loader (use GraphZSLDataset.load() normally)."""
    desc_maps = {
        "cora": CORA_DESCRIPTIONS,
        "citeseer": CITESEER_DESCRIPTIONS,
        "c-m10-m": CM10M_DESCRIPTIONS,
        "ogbn-arxiv": ARXIV_DESCRIPTIONS,
        "pubmed": PUBMED_DESCRIPTIONS,
        "wikics": WIKICS_DESCRIPTIONS,
        "amazon-computers": AMAZON_COMPUTERS_DESCRIPTIONS,
        "amazon-photo": AMAZON_PHOTO_DESCRIPTIONS,
        "coauthor-cs": COAUTHOR_CS_DESCRIPTIONS,
        "coauthor-physics": COAUTHOR_PHYSICS_DESCRIPTIONS,
    }
    name_lists = {
        "cora": list(CORA_DESCRIPTIONS.keys()),
        "citeseer": list(CITESEER_DESCRIPTIONS.keys()),
        "c-m10-m": list(CM10M_DESCRIPTIONS.keys()),
        "ogbn-arxiv": list(ARXIV_CATEGORY_NAMES),
        "pubmed": list(PUBMED_DESCRIPTIONS.keys()),
        "wikics": list(WIKICS_DESCRIPTIONS.keys()),
        "amazon-computers": list(AMAZON_COMPUTERS_DESCRIPTIONS.keys()),
        "amazon-photo": list(AMAZON_PHOTO_DESCRIPTIONS.keys()),
        "coauthor-cs": list(COAUTHOR_CS_DESCRIPTIONS.keys()),
        "coauthor-physics": list(COAUTHOR_PHYSICS_DESCRIPTIONS.keys()),
    }
    assert dataset_name in desc_maps, f"Unknown dataset: {dataset_name}"
    names = name_lists[dataset_name]
    descs = [desc_maps[dataset_name][n] for n in names]
    return encode_descriptions(descs, model_name)