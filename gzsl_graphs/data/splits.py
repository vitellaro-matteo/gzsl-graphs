"""
Fixed class split definitions for graph zero-shot learning.

Implements the seen/unseen class splits from DGPN (KDD 2021, Table 5, App. A.2).
DGPN Class Split II uses validation *classes* (not just validation nodes).

For new datasets (PubMed, WikiCS, Amazon Computers/Photo, Coauthor CS/Physics),
we define splits following the same convention: the first few class IDs are seen,
the middle ones are validation, and the last ones are unseen (test).

These are deterministic splits that ensure reproducibility across runs.
"""

from typing import Dict, List


CLASS_SPLITS = {
    "cora": {
        "class_split_1": {
            "train": [0, 1, 2],
            "test": [3, 4, 5, 6],
        },
        "class_split_2": {
            "train": [0, 1],
            "val": [2, 3],
            "test": [4, 5, 6],
        },
    },
    "citeseer": {
        "class_split_1": {
            "train": [0, 1],
            "test": [2, 3, 4, 5],
        },
        "class_split_2": {
            "train": [0, 1],
            "val": [2, 3],
            "test": [4, 5],
        },
    },
    "c-m10-m": {
        "class_split_1": {
            "train": [0, 1, 2],
            "test": [3, 4, 5],
        },
        "class_split_2": {
            "train": [0, 1],
            "val": [2, 3],
            "test": [4, 5],
        },
    },
    "ogbn-arxiv": {
        "class_split_2": {
            "train": list(range(0, 32)),
            "unseen": list(range(32, 40)),
        },
    },
    # =========================================================================
    # New datasets
    # =========================================================================
    "pubmed": {
        # PubMed: 3 classes (Experimental Diabetes, Type 1, Type 2)
        # With only 3 classes, splits are tight. Split 1: 1 seen / 2 unseen.
        # Split 2: 1 train / 1 val / 1 test.
        "class_split_1": {
            "train": [0],
            "test": [1, 2],
        },
        "class_split_2": {
            "train": [0],
            "val": [1],
            "test": [2],
        },
    },
    "wikics": {
        # WikiCS: 10 classes (CS sub-fields from Wikipedia)
        # 0: Computational Linguistics, 1: Databases, 2: Operating Systems,
        # 3: Computer Architecture, 4: Computer Security, 5: Internet Protocols,
        # 6: Computer File Systems, 7: Distributed Computing Architecture,
        # 8: Web Technology, 9: Programming Language Topics
        "class_split_1": {
            "train": [0, 1, 2, 3, 4],
            "test": [5, 6, 7, 8, 9],
        },
        "class_split_2": {
            "train": [0, 1, 2],
            "val": [3, 4, 5],
            "test": [6, 7, 8, 9],
        },
    },
    "amazon-computers": {
        # Amazon Computers: 10 classes (product categories)
        # 0: Desktops, 1: Data Storage, 2: Laptops, 3: Monitors,
        # 4: Computer Components, 5: Computer Accessories, 6: Networking Products,
        # 7: Tablets, 8: Servers, 9: Routers
        "class_split_1": {
            "train": [0, 1, 2, 3, 4],
            "test": [5, 6, 7, 8, 9],
        },
        "class_split_2": {
            "train": [0, 1, 2],
            "val": [3, 4, 5],
            "test": [6, 7, 8, 9],
        },
    },
    "amazon-photo": {
        # Amazon Photo: 8 classes (product categories)
        # 0: Digital Cameras, 1: Camera Lenses, 2: Tripods & Monopods,
        # 3: Flashes, 4: Camera Bags & Cases, 5: Video Surveillance,
        # 6: Lighting & Studio, 7: Binoculars & Scopes
        "class_split_1": {
            "train": [0, 1, 2, 3],
            "test": [4, 5, 6, 7],
        },
        "class_split_2": {
            "train": [0, 1],
            "val": [2, 3],
            "test": [4, 5, 6, 7],
        },
    },
    "coauthor-cs": {
        # Coauthor CS: 15 classes (fields of study in CS)
        # 0: Algorithms, 1: Artificial Intelligence, 2: Computer Vision,
        # 3: Databases, 4: Distributed Computing, 5: Graphics,
        # 6: HCI, 7: Information Retrieval, 8: Machine Learning,
        # 9: Natural Language Processing, 10: Networking,
        # 11: Operating Systems, 12: Programming Languages,
        # 13: Security, 14: Software Engineering
        "class_split_1": {
            "train": [0, 1, 2, 3, 4, 5, 6, 7],
            "test": [8, 9, 10, 11, 12, 13, 14],
        },
        "class_split_2": {
            "train": [0, 1, 2, 3, 4],
            "val": [5, 6, 7, 8, 9],
            "test": [10, 11, 12, 13, 14],
        },
    },
    "coauthor-physics": {
        # Coauthor Physics: 5 classes (fields of study in Physics)
        # 0: Condensed Matter, 1: High Energy Physics,
        # 2: Astrophysics, 3: Quantum Physics, 4: Nuclear Physics
        "class_split_1": {
            "train": [0, 1],
            "test": [2, 3, 4],
        },
        "class_split_2": {
            "train": [0, 1],
            "val": [2],
            "test": [3, 4],
        },
    },
    # =========================================================================
    # Extended datasets (added for large-scale benchmarking)
    # =========================================================================
    "cora-full": {
        # CoraFull: 70 classes (fine-grained CS research sub-topics)
        "class_split_1": {
            "train": list(range(0, 50)),
            "test": list(range(50, 70)),
        },
        "class_split_2": {
            "train": list(range(0, 42)),
            "val": list(range(42, 56)),
            "test": list(range(56, 70)),
        },
    },
    "ogbn-products": {
        # ogbn-products: 47 Amazon product categories (OGB, uses its own node splits)
        "class_split_2": {
            "train": list(range(0, 33)),
            "unseen": list(range(33, 47)),
        },
    },
    "reddit": {
        # Reddit2: 41 subreddit communities
        "class_split_1": {
            "train": list(range(0, 29)),
            "test": list(range(29, 41)),
        },
        "class_split_2": {
            "train": list(range(0, 23)),
            "val": list(range(23, 31)),
            "test": list(range(31, 41)),
        },
    },
    "roman-empire": {
        # Roman-empire: 18 grammatical/POS-tag classes
        "class_split_1": {
            "train": list(range(0, 12)),
            "test": list(range(12, 18)),
        },
        "class_split_2": {
            "train": list(range(0, 9)),
            "val": list(range(9, 14)),
            "test": list(range(14, 18)),
        },
    },
    "flickr": {
        # Flickr: 7 image category classes
        "class_split_1": {
            "train": [0, 1, 2, 3],
            "test": [4, 5, 6],
        },
        "class_split_2": {
            "train": [0, 1, 2],
            "val": [3],
            "test": [4, 5, 6],
        },
    },
    "lastfm-asia": {
        # LastFMAsia: 18 Asian country classes
        "class_split_1": {
            "train": list(range(0, 12)),
            "test": list(range(12, 18)),
        },
        "class_split_2": {
            "train": list(range(0, 9)),
            "val": list(range(9, 14)),
            "test": list(range(14, 18)),
        },
    },
    "actor": {
        # Actor: 5 film-actor type classes
        "class_split_1": {
            "train": [0, 1, 2],
            "test": [3, 4],
        },
        "class_split_2": {
            "train": [0, 1],
            "val": [2],
            "test": [3, 4],
        },
    },
    "chameleon": {
        # Chameleon (WikipediaNetwork): 5 monthly-traffic-level classes
        "class_split_1": {
            "train": [0, 1, 2],
            "test": [3, 4],
        },
        "class_split_2": {
            "train": [0, 1],
            "val": [2],
            "test": [3, 4],
        },
    },
    "squirrel": {
        # Squirrel (WikipediaNetwork): 5 monthly-traffic-level classes
        "class_split_1": {
            "train": [0, 1, 2],
            "test": [3, 4],
        },
        "class_split_2": {
            "train": [0, 1],
            "val": [2],
            "test": [3, 4],
        },
    },
    "amazon-ratings": {
        # Amazon-ratings: 5 star-rating classes (1 to 5 stars)
        "class_split_1": {
            "train": [0, 1, 2],
            "test": [3, 4],
        },
        "class_split_2": {
            "train": [0, 1],
            "val": [2],
            "test": [3, 4],
        },
    },
}


def get_class_split(dataset_name: str, split: str = "class_split_2") -> Dict[str, List[int]]:
    """Get the fixed class split for a dataset.

    Args:
        dataset_name: One of the 20 supported datasets — original:
                      'cora', 'citeseer', 'c-m10-m', 'ogbn-arxiv';
                      extended (PyG): 'pubmed', 'wikics', 'amazon-computers',
                      'amazon-photo', 'coauthor-cs', 'coauthor-physics',
                      'cora-full', 'reddit', 'roman-empire', 'flickr',
                      'lastfm-asia', 'actor', 'chameleon', 'squirrel',
                      'amazon-ratings'; OGB: 'ogbn-products'
        split: 'class_split_1' or 'class_split_2'

    Returns:
        Dict with 'train'/'val'/'test' (or 'train'/'unseen') mapping to class indices.
    """
    assert dataset_name in CLASS_SPLITS, f"Unknown dataset: {dataset_name}"
    assert split in CLASS_SPLITS[dataset_name], (
        f"Split '{split}' not available for {dataset_name}. "
        f"Available: {list(CLASS_SPLITS[dataset_name].keys())}"
    )
    return CLASS_SPLITS[dataset_name][split]