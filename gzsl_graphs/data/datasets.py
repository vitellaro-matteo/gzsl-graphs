"""
Unified dataset loader for graph zero-shot learning benchmarks.

Consolidates: ogbn_arxiv_loader_clean.py, cora_loader.py,
              citeseer_loader.py, cm10m_loader.py

This single loader is used by ALL models (DGPN, DBiGCN, ZeroG, ICIS),
ensuring fair comparison. Each dataset returns the same GraphZSLData format.

Supported datasets:
    - Original: cora, citeseer, c-m10-m, ogbn-arxiv
    - New:      pubmed, wikics, amazon-computers, amazon-photo,
                coauthor-cs, coauthor-physics

Usage:
    dataset = GraphZSLDataset("cora", root="./data")
    data = dataset.load()
    print(data.x.shape, data.seen_classes, data.class_semantics.shape)
"""

import torch
import numpy as np
import random
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .splits import get_class_split


@dataclass
class GraphZSLData:
    """Standardized graph data for zero-shot learning.

    All models receive this identical format, ensuring fair comparison.
    """
    x: torch.Tensor                                # [N, D] node features
    edge_index: torch.Tensor                       # [2, E] COO edges
    y: torch.Tensor                                # [N] node labels
    class_semantics: torch.Tensor                  # [C, S] CSD embeddings
    seen_classes: List[int]                        # seen class indices
    unseen_classes: List[int]                      # unseen class indices
    val_classes: List[int] = field(default_factory=list)
    train_mask: torch.Tensor = None                # seen-class train nodes
    val_mask: torch.Tensor = None                  # val-class nodes
    test_seen_mask: torch.Tensor = None            # seen-class test nodes
    test_unseen_mask: torch.Tensor = None          # unseen-class test nodes
    num_classes: int = 0
    class_names: Optional[List[str]] = None
    target_weights: Optional[torch.Tensor] = None  # [C, 2D] prototypes (ICIS)
    n_nodes: int = 0
    feature_dim: int = 0

    @property
    def test_mask(self):
        if self.test_seen_mask is not None and self.test_unseen_mask is not None:
            return self.test_seen_mask | self.test_unseen_mask
        return None

    # Backward-compat with old loader interface
    @property
    def seenclasses(self):
        return torch.LongTensor(self.seen_classes)

    @property
    def unseenclasses(self):
        return torch.LongTensor(self.unseen_classes)

    @property
    def attribute(self):
        return self.class_semantics

    @property
    def features(self):
        return self.x

    @property
    def labels(self):
        return self.y


class GraphZSLDataset:
    """Unified dataset loader for all graph ZSL benchmarks."""

    SUPPORTED = [
        "cora", "citeseer", "c-m10-m", "ogbn-arxiv",
        "pubmed", "wikics", "amazon-computers", "amazon-photo",
        "coauthor-cs", "coauthor-physics",
    ]

    def __init__(self, name, root="./data", split="class_split_2",
                 csd_type="text", semantic_model="all-MiniLM-L6-v2",
                 random_seed=0, test_ratio=0.2,
                 custom_seen_classes=None, custom_unseen_classes=None):
        assert name in self.SUPPORTED, f"Unknown dataset: {name}"
        self.name = name
        self.root = os.path.abspath(root)
        self.split = split
        self.csd_type = csd_type
        self.semantic_model = semantic_model
        self.random_seed = random_seed
        self.test_ratio = test_ratio
        self.custom_seen = custom_seen_classes
        self.custom_unseen = custom_unseen_classes

    def load(self) -> GraphZSLData:
        self._set_seeds()
        x, edge_index, y, n_classes, class_names = self._load_raw_graph()
        seen, unseen, val = self._get_split(n_classes)
        masks = self._create_node_masks(y, seen, unseen, val)
        class_semantics = self._create_class_semantics(x, y, n_classes, class_names)
        target_weights = self._compute_prototypes(x, y, n_classes)

        data = GraphZSLData(
            x=x, edge_index=edge_index, y=y,
            class_semantics=class_semantics,
            seen_classes=seen, unseen_classes=unseen, val_classes=val,
            train_mask=masks["train"], val_mask=masks["val"],
            test_seen_mask=masks["test_seen"], test_unseen_mask=masks["test_unseen"],
            num_classes=n_classes, class_names=class_names,
            target_weights=target_weights,
            n_nodes=x.size(0), feature_dim=x.size(1),
        )
        self._print_summary(data)
        return data

    # ---- Raw graph loading ----

    def _load_raw_graph(self):
        loaders = {
            "ogbn-arxiv": self._load_ogbn_arxiv,
            "cora": self._load_cora,
            "citeseer": self._load_citeseer,
            "c-m10-m": self._load_cm10m,
            "pubmed": self._load_pubmed,
            "wikics": self._load_wikics,
            "amazon-computers": self._load_amazon_computers,
            "amazon-photo": self._load_amazon_photo,
            "coauthor-cs": self._load_coauthor_cs,
            "coauthor-physics": self._load_coauthor_physics,
        }
        return loaders[self.name]()

    def _load_ogbn_arxiv(self):
        from ogb.nodeproppred import NodePropPredDataset
        os.environ["OGB_DATA_HOME"] = self.root
        dataset = NodePropPredDataset(name="ogbn-arxiv", root=self.root)
        self._ogb_dataset = dataset
        graph, labels = dataset[0]
        x = torch.FloatTensor(graph["node_feat"])
        edge_index = torch.LongTensor(graph["edge_index"])
        y = torch.LongTensor(labels.squeeze())
        n_classes = int(y.max().item()) + 1
        print(f"  ogbn-arXiv: {x.size(0)} nodes, {edge_index.size(1)} edges, {n_classes} classes")
        return x, edge_index, y, n_classes, list(ARXIV_CATEGORY_NAMES)

    def _load_cora(self):
        return self._load_planetoid("cora", 7, list(CORA_DESCRIPTIONS.keys()),
            {"Case_Based": 0, "Genetic_Algorithms": 1, "Neural_Networks": 2,
             "Probabilistic_Methods": 3, "Reinforcement_Learning": 4,
             "Rule_Learning": 5, "Theory": 6})

    def _load_citeseer(self):
        return self._load_planetoid("citeseer", 6, list(CITESEER_DESCRIPTIONS.keys()),
            {"Agents": 0, "AI": 1, "DB": 2, "IR": 3, "ML": 4, "HCI": 5})

    def _load_planetoid(self, name, n_classes, class_names, label_map):
        data_dir = Path(self.root) / name
        content_file = next(data_dir.glob("*.content"), None)
        cites_file = next(data_dir.glob("*.cites"), None)
        if not content_file or not cites_file:
            raise FileNotFoundError(f"Need .content and .cites in {data_dir}")

        node_ids, features, labels_str = [], [], []
        with open(content_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                node_ids.append(parts[0])
                features.append([int(v) for v in parts[1:-1]])
                labels_str.append(parts[-1])

        node_map = {nid: idx for idx, nid in enumerate(node_ids)}
        x = torch.FloatTensor(np.array(features, dtype=np.float32))
        y = torch.LongTensor([label_map[l] for l in labels_str])

        edges = []
        with open(cites_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2: continue
                if parts[0] in node_map and parts[1] in node_map:
                    edges.append([node_map[parts[0]], node_map[parts[1]]])
        edges_np = np.array(edges)
        all_edges = np.unique(np.vstack([edges_np, edges_np[:, [1, 0]]]), axis=0)
        edge_index = torch.LongTensor(all_edges.T)

        print(f"  {name}: {x.size(0)} nodes, {edge_index.size(1)} edges, {n_classes} classes")
        return x, edge_index, y, n_classes, class_names

    def _load_cm10m(self):
        data_dir = Path(self.root) / "C-M10-M"
        node_features = {}
        with open(data_dir / "feature.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                node_features[int(float(parts[0]))] = [float(v) for v in parts[1:]]
        sorted_ids = sorted(node_features.keys())
        node_map = {orig: idx for idx, orig in enumerate(sorted_ids)}
        n_nodes, feat_dim = len(node_map), len(node_features[sorted_ids[0]])
        x = torch.zeros(n_nodes, feat_dim)
        for orig_id, feats in node_features.items():
            x[node_map[orig_id]] = torch.tensor(feats)

        edges = []
        with open(data_dir / "graph.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2: continue
                src, dst = int(float(parts[0])), int(float(parts[1]))
                if src in node_map and dst in node_map and src != dst:
                    edges.append([node_map[src], node_map[dst]])
        edges_np = np.array(edges)
        all_edges = np.unique(np.vstack([edges_np, edges_np[:, [1, 0]]]), axis=0)
        edge_index = torch.LongTensor(all_edges.T)

        y = torch.zeros(n_nodes, dtype=torch.long)
        with open(data_dir / "group.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2: continue
                nid, cid = int(float(parts[0])), int(parts[1])
                if nid in node_map: y[node_map[nid]] = cid

        print(f"  C-M10-M: {n_nodes} nodes, {edge_index.size(1)} edges, 6 classes")
        return x, edge_index, y, 6, list(CM10M_DESCRIPTIONS.keys())

    # ---- New dataset loaders (PyTorch Geometric) ----

    def _load_pubmed(self):
        """Load PubMed citation network via PyTorch Geometric Planetoid."""
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root=self.root, name="PubMed")
        self._pyg_dataset = dataset
        data = dataset[0]
        x = data.x
        edge_index = data.edge_index
        y = data.y
        n_classes = int(y.max().item()) + 1
        class_names = list(PUBMED_DESCRIPTIONS.keys())
        print(f"  PubMed: {x.size(0)} nodes, {edge_index.size(1)} edges, {n_classes} classes")
        return x, edge_index, y, n_classes, class_names

    def _load_wikics(self):
        """Load WikiCS dataset via PyTorch Geometric."""
        from torch_geometric.datasets import WikiCS
        dataset = WikiCS(root=os.path.join(self.root, "WikiCS"), is_undirected=True)
        self._pyg_dataset = dataset
        data = dataset[0]
        x = data.x
        edge_index = data.edge_index
        y = data.y
        n_classes = int(y.max().item()) + 1
        class_names = list(WIKICS_DESCRIPTIONS.keys())
        print(f"  WikiCS: {x.size(0)} nodes, {edge_index.size(1)} edges, {n_classes} classes")
        return x, edge_index, y, n_classes, class_names

    def _load_amazon_computers(self):
        """Load Amazon Computers co-purchase graph via PyTorch Geometric."""
        from torch_geometric.datasets import Amazon
        dataset = Amazon(root=os.path.join(self.root, "Amazon"), name="Computers")
        self._pyg_dataset = dataset
        data = dataset[0]
        x = data.x
        edge_index = data.edge_index
        y = data.y
        n_classes = int(y.max().item()) + 1
        class_names = list(AMAZON_COMPUTERS_DESCRIPTIONS.keys())
        assert n_classes == len(class_names), (
            f"Amazon Computers: expected {len(class_names)} classes, got {n_classes}. "
            f"Class name list may need updating."
        )
        print(f"  Amazon Computers: {x.size(0)} nodes, {edge_index.size(1)} edges, {n_classes} classes")
        return x, edge_index, y, n_classes, class_names

    def _load_amazon_photo(self):
        """Load Amazon Photo co-purchase graph via PyTorch Geometric."""
        from torch_geometric.datasets import Amazon
        dataset = Amazon(root=os.path.join(self.root, "Amazon"), name="Photo")
        self._pyg_dataset = dataset
        data = dataset[0]
        x = data.x
        edge_index = data.edge_index
        y = data.y
        n_classes = int(y.max().item()) + 1
        class_names = list(AMAZON_PHOTO_DESCRIPTIONS.keys())
        assert n_classes == len(class_names), (
            f"Amazon Photo: expected {len(class_names)} classes, got {n_classes}. "
            f"Class name list may need updating."
        )
        print(f"  Amazon Photo: {x.size(0)} nodes, {edge_index.size(1)} edges, {n_classes} classes")
        return x, edge_index, y, n_classes, class_names

    def _load_coauthor_cs(self):
        """Load Coauthor CS co-authorship graph via PyTorch Geometric."""
        from torch_geometric.datasets import Coauthor
        dataset = Coauthor(root=os.path.join(self.root, "Coauthor"), name="CS")
        self._pyg_dataset = dataset
        data = dataset[0]
        x = data.x
        edge_index = data.edge_index
        y = data.y
        n_classes = int(y.max().item()) + 1
        class_names = list(COAUTHOR_CS_DESCRIPTIONS.keys())
        assert n_classes == len(class_names), (
            f"Coauthor CS: expected {len(class_names)} classes, got {n_classes}. "
            f"Class name list may need updating."
        )
        print(f"  Coauthor CS: {x.size(0)} nodes, {edge_index.size(1)} edges, {n_classes} classes")
        return x, edge_index, y, n_classes, class_names

    def _load_coauthor_physics(self):
        """Load Coauthor Physics co-authorship graph via PyTorch Geometric."""
        from torch_geometric.datasets import Coauthor
        dataset = Coauthor(root=os.path.join(self.root, "Coauthor"), name="Physics")
        self._pyg_dataset = dataset
        data = dataset[0]
        x = data.x
        edge_index = data.edge_index
        y = data.y
        n_classes = int(y.max().item()) + 1
        class_names = list(COAUTHOR_PHYSICS_DESCRIPTIONS.keys())
        assert n_classes == len(class_names), (
            f"Coauthor Physics: expected {len(class_names)} classes, got {n_classes}. "
            f"Class name list may need updating."
        )
        print(f"  Coauthor Physics: {x.size(0)} nodes, {edge_index.size(1)} edges, {n_classes} classes")
        return x, edge_index, y, n_classes, class_names

    # ---- Class splits ----

    def _get_split(self, n_classes):
        if self.custom_seen is not None and self.custom_unseen is not None:
            return sorted(self.custom_seen), sorted(self.custom_unseen), []
        split_def = get_class_split(self.name, self.split)
        if self.name == "ogbn-arxiv":
            return split_def["train"], split_def["unseen"], []
        return split_def["train"], split_def["test"], split_def.get("val", [])

    # ---- Node masks ----

    def _create_node_masks(self, y, seen, unseen, val_classes):
        if self.name == "ogbn-arxiv":
            return self._ogbn_masks(y, set(seen), set(unseen))
        # PubMed loaded via Planetoid has built-in splits we could use,
        # but for GZSL consistency we use the same class-based splitting
        # as all other citation datasets.
        return self._citation_masks(y, set(seen), set(unseen), set(val_classes))

    def _ogbn_masks(self, y, seen_set, unseen_set):
        split_idx = self._ogb_dataset.get_idx_split()
        n = y.size(0)
        is_seen = torch.zeros(n, dtype=torch.bool)
        is_unseen = torch.zeros(n, dtype=torch.bool)
        for c in seen_set: is_seen |= (y == c)
        for c in unseen_set: is_unseen |= (y == c)
        ogb_train = torch.zeros(n, dtype=torch.bool); ogb_train[split_idx["train"]] = True
        ogb_val = torch.zeros(n, dtype=torch.bool); ogb_val[split_idx["valid"]] = True
        ogb_test = torch.zeros(n, dtype=torch.bool); ogb_test[split_idx["test"]] = True
        return {"train": ogb_train & is_seen, "val": ogb_val & is_seen,
                "test_seen": ogb_test & is_seen, "test_unseen": ogb_test & is_unseen}

    def _citation_masks(self, y, seen_set, unseen_set, val_set):
        n = y.size(0)
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_seen = torch.zeros(n, dtype=torch.bool)
        test_unseen = torch.zeros(n, dtype=torch.bool)
        for c in seen_set:
            idx = torch.where(y == c)[0]
            perm = idx[torch.randperm(len(idx))]
            n_test = max(1, int(len(idx) * self.test_ratio))
            train_mask[perm[n_test:]] = True
            test_seen[perm[:n_test]] = True
        for c in val_set: val_mask[y == c] = True
        for c in unseen_set: test_unseen[y == c] = True
        return {"train": train_mask, "val": val_mask,
                "test_seen": test_seen, "test_unseen": test_unseen}

    # ---- Semantic embeddings ----

    def _create_class_semantics(self, x, y, n_classes, class_names):
        if self.csd_type == "statistical":
            return self._statistical_embeddings(x, y, n_classes)
        descs = self._get_class_descriptions(class_names)
        return self._encode_descriptions(descs)

    def _encode_descriptions(self, descriptions):
        from sentence_transformers import SentenceTransformer
        print(f"  Encoding {len(descriptions)} descriptions ({self.semantic_model})...")
        model = SentenceTransformer(self.semantic_model)
        emb = model.encode(descriptions, convert_to_tensor=True)
        emb = emb / emb.norm(dim=1, keepdim=True)
        print(f"  Semantic embeddings: {emb.shape}")
        return emb

    def _statistical_embeddings(self, x, y, n_classes):
        embs = []
        for c in range(n_classes):
            feats = x[y == c]
            embs.append(torch.cat([feats.mean(0), feats.std(0)]) if len(feats) > 0
                        else torch.zeros(x.size(1) * 2))
        result = torch.stack(embs)
        return result / (result.norm(dim=1, keepdim=True) + 1e-8)

    def _get_class_descriptions(self, class_names):
        all_descs = {
            **CORA_DESCRIPTIONS, **CITESEER_DESCRIPTIONS,
            **CM10M_DESCRIPTIONS, **ARXIV_DESCRIPTIONS,
            **PUBMED_DESCRIPTIONS, **WIKICS_DESCRIPTIONS,
            **AMAZON_COMPUTERS_DESCRIPTIONS, **AMAZON_PHOTO_DESCRIPTIONS,
            **COAUTHOR_CS_DESCRIPTIONS, **COAUTHOR_PHYSICS_DESCRIPTIONS,
        }
        return [all_descs.get(n, f"Research area: {n}") for n in class_names]

    # ---- Prototypes ----

    def _compute_prototypes(self, x, y, n_classes):
        protos = torch.zeros(n_classes, x.size(1))
        stds = torch.zeros(n_classes, x.size(1))
        for c in range(n_classes):
            feats = x[y == c]
            if len(feats) > 0:
                protos[c] = feats.mean(0); stds[c] = feats.std(0)
        weights = torch.cat([protos, stds], dim=1)
        norm = weights.norm(dim=1).mean()
        return weights * (2.5 / norm) if norm > 0 else weights

    # ---- Helpers ----

    def _set_seeds(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def _print_summary(self, data):
        s = "=" * 60
        lines = [
            f"\n{s}", f"  Dataset:        {self.name}",
            f"  Nodes:          {data.n_nodes}",
            f"  Edges:          {data.edge_index.size(1)}",
            f"  Features:       {data.feature_dim}",
            f"  Classes:        {data.num_classes}",
            f"  Seen classes:   {data.seen_classes}",
            f"  Unseen classes: {data.unseen_classes}",
        ]
        if data.val_classes: lines.append(f"  Val classes:    {data.val_classes}")
        lines += [
            f"  Semantics dim:  {data.class_semantics.size(1)}",
            f"  Train nodes:    {data.train_mask.sum().item()}",
        ]
        if data.val_mask is not None:
            lines.append(f"  Val nodes:      {data.val_mask.sum().item()}")
        lines += [
            f"  Test seen:      {data.test_seen_mask.sum().item()}",
            f"  Test unseen:    {data.test_unseen_mask.sum().item()}",
            s,
        ]
        print("\n".join(lines))


# ============================================================
# Class Descriptions (original datasets)
# ============================================================

CORA_DESCRIPTIONS = {
    "Case_Based": "Case-Based Reasoning uses past experiences and specific cases to solve new problems by finding similar situations and adapting previous solutions",
    "Genetic_Algorithms": "Genetic Algorithms are evolutionary optimization techniques inspired by natural selection that evolve solutions through selection, crossover, and mutation",
    "Neural_Networks": "Neural Networks are computational models inspired by biological neurons that learn patterns through interconnected layers of artificial neurons",
    "Probabilistic_Methods": "Probabilistic Methods use probability theory and statistical inference to handle uncertainty and make predictions from data",
    "Reinforcement_Learning": "Reinforcement Learning trains agents to make sequences of decisions by learning from rewards and penalties in an environment",
    "Rule_Learning": "Rule Learning discovers logical rules and decision trees from data to make classifications and predictions",
    "Theory": "Theory focuses on theoretical computer science including computational complexity, algorithms, and formal methods",
}

CITESEER_DESCRIPTIONS = {
    "Agents": "Agents and Multi-Agent Systems study autonomous software agents that perceive their environment and take actions to achieve goals through cooperation and coordination",
    "AI": "Artificial Intelligence develops intelligent systems capable of reasoning, learning, perception, and problem-solving using symbolic and sub-symbolic approaches",
    "DB": "Databases focus on structured data storage, querying, indexing, and transaction processing for efficient data management and retrieval",
    "IR": "Information Retrieval develops systems for searching, ranking, and retrieving relevant information from large document collections using text analysis and relevance models",
    "ML": "Machine Learning creates algorithms that automatically learn patterns from data and improve performance through experience without explicit programming",
    "HCI": "Human-Computer Interaction studies the design and evaluation of interactive systems, user interfaces, and the relationships between humans and computers",
}

CM10M_DESCRIPTIONS = {
    "Biology": "Biology and biological sciences encompassing molecular biology, genetics, ecology, and life sciences research",
    "Computer Science": "Computer Science including algorithms, machine learning, software engineering, and computational theory",
    "Financial Economics": "Financial Economics covering econometrics, market analysis, financial modeling, and economic theory",
    "Industrial Engineering": "Industrial Engineering focusing on operations research, optimization, manufacturing systems, and process improvement",
    "Physics": "Physics including theoretical physics, experimental physics, quantum mechanics, and physical phenomena",
    "Social Science": "Social Science encompassing sociology, psychology, political science, and human behavior research",
}

ARXIV_DESCRIPTIONS = {
    "cs.AI": "Artificial Intelligence focuses on creating intelligent agents and systems",
    "cs.AR": "Hardware Architecture deals with computer system design and organization",
    "cs.CC": "Computational Complexity studies the resources needed to solve problems",
    "cs.CE": "Computational Engineering applies computational methods to engineering",
    "cs.CG": "Computational Geometry focuses on algorithms for geometric problems",
    "cs.CL": "Computation and Language processes natural language understanding",
    "cs.CR": "Cryptography and Security protects information and communications",
    "cs.CV": "Computer Vision enables machines to interpret visual information",
    "cs.CY": "Computers and Society examines social implications of computing",
    "cs.DB": "Databases manages structured data storage and retrieval",
    "cs.DC": "Distributed Computing coordinates multiple computers working together",
    "cs.DL": "Digital Libraries organizes and provides access to digital collections",
    "cs.DM": "Discrete Mathematics studies discrete mathematical structures",
    "cs.DS": "Data Structures and Algorithms designs efficient data organization",
    "cs.ET": "Emerging Technologies explores new computing paradigms",
    "cs.FL": "Formal Languages and Automata studies computational models",
    "cs.GL": "General Literature covers broad computer science topics",
    "cs.GR": "Graphics creates and manipulates visual content",
    "cs.GT": "Computer Science and Game Theory applies game theory to computing",
    "cs.HC": "Human-Computer Interaction designs user interfaces and experiences",
    "cs.IR": "Information Retrieval finds relevant information in large collections",
    "cs.IT": "Information Theory quantifies and processes information",
    "cs.LG": "Machine Learning develops algorithms that learn from data",
    "cs.LO": "Logic in Computer Science applies formal logic to computing",
    "cs.MA": "Multiagent Systems coordinates multiple autonomous agents",
    "cs.MM": "Multimedia combines multiple forms of media content",
    "cs.MS": "Mathematical Software develops numerical and symbolic computation",
    "cs.NA": "Numerical Analysis studies algorithms for continuous problems",
    "cs.NE": "Neural and Evolutionary Computing uses bio-inspired algorithms",
    "cs.NI": "Networking and Internet Architecture designs communication systems",
    "cs.OH": "Other Computer Science covers miscellaneous topics",
    "cs.OS": "Operating Systems manages computer hardware and software resources",
    "cs.PF": "Performance measures and optimizes system efficiency",
    "cs.PL": "Programming Languages designs languages for software development",
    "cs.RO": "Robotics creates autonomous physical agents",
    "cs.SC": "Symbolic Computation manipulates mathematical expressions",
    "cs.SD": "Sound processes and synthesizes audio signals",
    "cs.SE": "Software Engineering develops large-scale software systems",
    "cs.SI": "Social and Information Networks analyzes networked systems",
    "cs.SY": "Systems and Control designs and analyzes control systems",
}

ARXIV_CATEGORY_NAMES = [
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV",
    "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL",
    "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO",
    "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS",
    "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY",
]


# ============================================================
# Class Descriptions (new datasets)
# ============================================================

# PubMed: 3 classes — diabetes-related citation network
# Class labels: 0=Experimental Diabetes, 1=Type 1, 2=Type 2
# Descriptions adapted from ZeroG (KDD 2024, App. B.2.3)
PUBMED_DESCRIPTIONS = {
    "Experimental Diabetes": (
        "Experimental studies on diabetes mechanisms and therapies, including "
        "laboratory research on disease models, pathophysiology, and novel "
        "treatment approaches for understanding diabetes at a molecular level"
    ),
    "Type 1 Diabetes": (
        "Type 1 Diabetes research focusing on autoimmune processes and treatments, "
        "including studies on immune system dysfunction, beta cell destruction, "
        "insulin therapy, and immunological interventions"
    ),
    "Type 2 Diabetes": (
        "Type 2 Diabetes studies emphasizing insulin resistance and management "
        "strategies, including research on metabolic syndrome, lifestyle "
        "interventions, oral medications, and long-term disease management"
    ),
}

# WikiCS: 10 classes — Wikipedia articles on CS sub-fields
# Class labels 0-9 as documented in ZeroG (KDD 2024, App. B.2.7)
WIKICS_DESCRIPTIONS = {
    "Computational Linguistics": (
        "Computational Linguistics focuses on the intersection of computer science "
        "and linguistics, including natural language processing, machine translation, "
        "speech recognition, and computational models of language"
    ),
    "Databases": (
        "Databases covers database technologies and theories, including relational "
        "databases, query optimization, data modeling, transaction processing, "
        "and distributed data management systems"
    ),
    "Operating Systems": (
        "Operating Systems details the software that manages computer hardware, "
        "including process scheduling, memory management, file systems, "
        "device drivers, and system security"
    ),
    "Computer Architecture": (
        "Computer Architecture explores the design and structure of computer "
        "systems, including processor design, instruction sets, memory hierarchy, "
        "pipelining, and parallel computing architectures"
    ),
    "Computer Security": (
        "Computer Security addresses the protection of information systems, "
        "including cryptography, network security, access control, malware "
        "analysis, and secure software development"
    ),
    "Internet Protocols": (
        "Internet Protocols discusses the rules governing internet data exchange, "
        "including TCP/IP, HTTP, DNS, routing protocols, and network layer "
        "communication standards"
    ),
    "Computer File Systems": (
        "Computer File Systems covers methods for storing and organizing computer "
        "files, including file system design, storage allocation, journaling, "
        "and distributed file systems"
    ),
    "Distributed Computing Architecture": (
        "Distributed Computing Architecture concerns computations spread across "
        "multiple machines, including cloud computing, peer-to-peer systems, "
        "consensus algorithms, and fault tolerance"
    ),
    "Web Technology": (
        "Web Technology focuses on the technologies underpinning the web, "
        "including web standards, client-server architecture, web frameworks, "
        "semantic web, and web services"
    ),
    "Programming Language Topics": (
        "Programming Language Topics includes various aspects of programming "
        "languages, including language design, type systems, compilers, "
        "runtime systems, and programming paradigms"
    ),
}

# Amazon Computers: 10 classes — product categories in co-purchase graph
# NOTE: The Shchur et al. dataset uses integer labels without official names.
# These names are inferred from the Amazon product hierarchy for the
# electronics/computers segment. If the actual class count differs at load
# time, the assertion in _load_amazon_computers will catch it.
AMAZON_COMPUTERS_DESCRIPTIONS = {
    "Desktops": (
        "Desktop computers and workstations including tower PCs, all-in-one "
        "computers, and mini PCs for personal and professional use"
    ),
    "Data Storage": (
        "Data storage devices including hard drives, solid-state drives, USB "
        "flash drives, memory cards, and external storage solutions"
    ),
    "Laptops": (
        "Laptop computers and notebooks including ultrabooks, gaming laptops, "
        "and portable workstations for mobile computing"
    ),
    "Monitors": (
        "Computer monitors and displays including LCD, LED, and OLED screens "
        "for desktop computing, gaming, and professional graphics work"
    ),
    "Computer Components": (
        "Internal computer components including processors, graphics cards, "
        "motherboards, RAM modules, and power supplies for building and "
        "upgrading computers"
    ),
    "Computer Accessories": (
        "Computer peripherals and accessories including keyboards, mice, "
        "webcams, speakers, and other input/output devices"
    ),
    "Networking Products": (
        "Networking equipment including switches, hubs, network adapters, "
        "cables, and other connectivity products for wired and wireless networks"
    ),
    "Tablets": (
        "Tablet computers and e-readers including touchscreen portable devices "
        "and their accessories for mobile computing and content consumption"
    ),
    "Servers": (
        "Server hardware and components including rack servers, server "
        "processors, and enterprise storage for data center operations"
    ),
    "Routers": (
        "Network routers and modems including wireless routers, mesh systems, "
        "and gateway devices for home and business network connectivity"
    ),
}

# Amazon Photo: 8 classes — product categories in photography co-purchase graph
# NOTE: Same caveat as Amazon Computers — names inferred from domain.
AMAZON_PHOTO_DESCRIPTIONS = {
    "Digital Cameras": (
        "Digital cameras including DSLR, mirrorless, compact, and action cameras "
        "for photography and videography"
    ),
    "Camera Lenses": (
        "Camera lenses and lens accessories including zoom lenses, prime lenses, "
        "macro lenses, and wide-angle lenses for various photography styles"
    ),
    "Tripods and Monopods": (
        "Camera support equipment including tripods, monopods, stabilizers, "
        "and gimbals for steady photography and video recording"
    ),
    "Flashes": (
        "Camera flash units and lighting accessories including speedlights, "
        "flash diffusers, and trigger systems for controlled illumination"
    ),
    "Camera Bags and Cases": (
        "Protective camera bags, cases, and carrying solutions including "
        "backpacks, shoulder bags, and hard cases for equipment transport"
    ),
    "Video Surveillance": (
        "Video surveillance and security camera systems including IP cameras, "
        "DVR systems, and monitoring equipment for security applications"
    ),
    "Lighting and Studio": (
        "Photography lighting and studio equipment including continuous lights, "
        "softboxes, reflectors, and backdrops for professional studio setups"
    ),
    "Binoculars and Scopes": (
        "Optical instruments including binoculars, telescopes, spotting scopes, "
        "and rangefinders for observation and outdoor activities"
    ),
}

# Coauthor CS: 15 classes — fields of study (CS sub-fields from MAG)
# NOTE: The Shchur et al. dataset derives labels from Microsoft Academic Graph.
# These are the most active research field labels for CS authors.
COAUTHOR_CS_DESCRIPTIONS = {
    "Algorithms": (
        "Algorithms and computational theory including algorithm design, "
        "complexity analysis, graph algorithms, and combinatorial optimization"
    ),
    "Artificial Intelligence": (
        "Artificial Intelligence including knowledge representation, reasoning, "
        "expert systems, planning, and general AI methodologies"
    ),
    "Computer Vision": (
        "Computer Vision including image recognition, object detection, scene "
        "understanding, video analysis, and visual perception systems"
    ),
    "Databases": (
        "Database systems and data management including query processing, "
        "data mining, data warehousing, and information extraction"
    ),
    "Distributed Computing": (
        "Distributed computing and parallel systems including cloud computing, "
        "distributed algorithms, middleware, and scalable architectures"
    ),
    "Graphics": (
        "Computer graphics and visualization including rendering, geometric "
        "modeling, animation, virtual reality, and scientific visualization"
    ),
    "Human-Computer Interaction": (
        "Human-Computer Interaction including user interface design, usability "
        "engineering, interaction techniques, and user experience research"
    ),
    "Information Retrieval": (
        "Information retrieval and web search including text mining, "
        "recommendation systems, question answering, and search engines"
    ),
    "Machine Learning": (
        "Machine Learning including supervised learning, unsupervised learning, "
        "deep learning, reinforcement learning, and statistical learning theory"
    ),
    "Natural Language Processing": (
        "Natural Language Processing including text analysis, sentiment analysis, "
        "machine translation, summarization, and dialogue systems"
    ),
    "Networking": (
        "Computer networking including network protocols, wireless networks, "
        "mobile computing, Internet of Things, and network security"
    ),
    "Operating Systems": (
        "Operating systems and systems software including process management, "
        "file systems, virtualization, and embedded systems"
    ),
    "Programming Languages": (
        "Programming languages including compiler design, type theory, "
        "program analysis, software verification, and language implementation"
    ),
    "Security": (
        "Computer security and cryptography including network security, "
        "privacy, access control, malware analysis, and applied cryptography"
    ),
    "Software Engineering": (
        "Software engineering including software design, testing, maintenance, "
        "software architecture, and development methodologies"
    ),
}

# Coauthor Physics: 5 classes — fields of study (Physics sub-fields from MAG)
COAUTHOR_PHYSICS_DESCRIPTIONS = {
    "Condensed Matter": (
        "Condensed matter physics including solid state physics, superconductivity, "
        "magnetism, semiconductor physics, and materials science"
    ),
    "High Energy Physics": (
        "High energy physics and particle physics including quantum field theory, "
        "the standard model, collider experiments, and fundamental interactions"
    ),
    "Astrophysics": (
        "Astrophysics including stellar physics, cosmology, galaxy formation, "
        "gravitational waves, and observational astronomy"
    ),
    "Quantum Physics": (
        "Quantum physics including quantum information, quantum computing, "
        "quantum optics, entanglement, and foundations of quantum mechanics"
    ),
    "Nuclear Physics": (
        "Nuclear physics including nuclear structure, nuclear reactions, "
        "radioactivity, nuclear energy, and nuclear astrophysics"
    ),
}