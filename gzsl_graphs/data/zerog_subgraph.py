"""
k-hop subgraph dataset for ZeroG training.

Extracts subgraphs centered at each node, filtered by class diversity.
Each subgraph must contain at least ceil(C/2) distinct classes and
at most max_nodes nodes. This produces the training data for ZeroG's
prompt-based subgraph sampling.

Cleaned from: SubgraphDataset.py (original ZeroG repo)
"""

import math
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_dense_adj, k_hop_subgraph
from tqdm import tqdm


# Dataset-specific min_classes overrides from original ZeroG code.
# Datasets not listed here use the default: ceil(num_unique_classes / 2).
# Rationale: datasets with very few classes (e.g. PubMed=3) or very many
# classes (e.g. Coauthor CS=15) need tuned thresholds to produce a
# reasonable number of valid subgraphs.
_MIN_CLASSES_OVERRIDES = {
    "Citeseer": 2,
    "Arxiv": 5,
    # New datasets — conservative defaults to avoid empty subgraph sets.
    # PubMed has only 3 classes; ceil(3/2)=2 is fine, no override needed.
    # WikiCS has 10 classes; ceil(10/2)=5 may be too strict for sparse graph.
    "WikiCS": 3,
    # Amazon graphs are denser (avg degree ~26-36), ceil(C/2) is feasible.
    # Coauthor CS has 15 classes; ceil(15/2)=8 is very strict for 2-hop.
    "Coauthor-CS": 4,
    # Coauthor Physics has 5 classes; ceil(5/2)=3 is fine, no override needed.
}


class kHopSubgraphDataset(Dataset):
    """Pre-computed k-hop subgraph dataset for ZeroG training.

    Extracts all valid subgraphs at initialization time (cached in memory).

    Args:
        data: PyG Data object with .edge_index, .y, .raw_texts, .label_text/.label_name
        num_hops: Number of hops for subgraph extraction (default: 2)
        max_nodes: Maximum nodes per subgraph (default: 100)
        dataset_name: Name string stored on each subgraph for prompt selection
        min_classes: Override minimum class diversity (default: ceil(C/2) or
                     dataset-specific override)
    """

    def __init__(self, data, num_hops=2, max_nodes=100, dataset_name="Cora",
                 min_classes=None):
        super().__init__(None, None, None)
        self.data = data
        self.num_hops = num_hops
        self.max_nodes = max_nodes
        self.dataset_name = dataset_name

        unique_classes = data.y.unique()
        if min_classes is not None:
            self.min_classes = min_classes
        elif dataset_name in _MIN_CLASSES_OVERRIDES:
            self.min_classes = _MIN_CLASSES_OVERRIDES[dataset_name]
        else:
            self.min_classes = math.ceil(len(unique_classes) / 2)

        self.subgraphs = self._create_subgraphs()

    def _create_subgraphs(self):
        subgraphs = []
        label_text = (self.data.label_text
                      if hasattr(self.data, "label_text")
                      else self.data.label_name)

        for idx in tqdm(range(self.data.num_nodes),
                        desc=f"Extracting {self.dataset_name} subgraphs"):
            sg_nodes, sg_edges, mapping, _ = k_hop_subgraph(
                node_idx=idx,
                num_hops=self.num_hops,
                edge_index=self.data.edge_index,
                relabel_nodes=True,
                num_nodes=self.data.num_nodes,
            )
            unique_cls = np.unique(self.data.y[sg_nodes].cpu().numpy())
            if (len(unique_cls) >= self.min_classes
                    and len(sg_nodes) <= self.max_nodes):
                sub = Data(edge_index=sg_edges)
                sub.y = self.data.y[sg_nodes]
                sub.raw_text = [self.data.raw_texts[i] for i in sg_nodes.tolist()]
                sub.label_text = list(label_text)
                sub.dataset_name = self.dataset_name
                subgraphs.append(sub)

        print(f"  {self.dataset_name}: {len(subgraphs)} valid subgraphs "
              f"(of {self.data.num_nodes} nodes, min_classes={self.min_classes})")
        return subgraphs

    def len(self):
        return len(self.subgraphs)

    def get(self, idx):
        return self.subgraphs[idx]

    def __getitem__(self, idx):
        return self.get(idx)