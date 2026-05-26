"""
ZeroG: Cross-dataset Zero-shot Transferability in Graphs.

Li et al., KDD 2024. https://github.com/NineAbyss/ZeroG

Architecture (faithful to original):
  - SentenceBERT (multi-qa-distilbert-cos-v1) with LoRA fine-tuning
  - Training: extract k-hop subgraphs, add virtual prompt node with
    dataset description, encode text via LM, do R rounds of neighbor
    aggregation, compute cross-entropy with label embeddings
  - Inference: encode all nodes + labels, R rounds propagation,
    cosine similarity -> argmax

Key differences from DGPN/DBiGCN:
  - Cross-dataset: trains on source graphs, tests on different target graphs
  - Uses raw text (titles/abstracts) not pre-computed features
  - Requires .pt dataset files with raw_texts and label_name fields
  - Only LoRA parameters are trained (~70K params)

This file contains the model only. Training logic in scripts/train_zerog.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import LoraModel, LoraConfig
from sklearn.metrics import accuracy_score


# Dataset descriptions used as virtual prompt nodes (from original repo)
DATASET_DESCRIPTIONS = {
    "Cora": "The Cora dataset is a fundamental resource in the field of graph learning, particularly within the realm of machine learning research. It represents a network of scientific publications. There are 7 categories in Cora: Theory: This category covers theoretical aspects of machine learning and AI. Reinforcement Learning: This category includes research on reinforcement learning, a type of machine learning where an agent learns to make decisions to achieve a goal, focusing on algorithms, methodologies, and applications in decision-making areas. Genetic Algorithms: This category deals with genetic algorithms, a type of optimization algorithm inspired by natural evolution. Neural Networks: This category focuses on artificial neural networks, a subset of machine learning mimicking the human brain, covering various architectures, training techniques, and applications. Probabilistic Methods: This category pertains to research on probabilistic methods in machine learning, using probability mathematics to handle uncertainty and make predictions. Case Based: This category focuses on case-based reasoning in AI, a method that solves new problems by referring to similar past cases. Rule Learning: This category is about rule-based learning in machine learning, involving the generation of rules for decision-making systems, focusing on algorithms, transparency, and applications in fields requiring interpretability. The average degree of Cora is 4.",
    "Citeseer": "The Citeseer dataset is a prominent academic resource in the field of computer science, categorizing publications into six distinct areas. These are Agents, focusing on intelligent agents; Machine Learning (ML), covering all aspects of learning techniques and applications; Information Retrieval (IR), dealing with data and text indexing and retrieval; Databases (DB), related to database management and data mining; Human-Computer Interaction (HCI), emphasizing computer technology interfaces for humans; and Artificial Intelligence (AI), a broad category encompassing general AI theory and applications, excluding certain subfields. The average degree of this graph is 2.",
    "Pubmed": "The PubMed dataset comprises three categories: Experimental studies on diabetes mechanisms and therapies, Type 1 Diabetes research focusing on autoimmune processes and treatments, and Type 2 Diabetes studies emphasizing insulin resistance and management strategies. Each category addresses specific aspects of diabetes research, aiding in understanding and treating this complex disease. The average degree of this graph is 4.5.",
    "Arxiv": "The arXiv dataset is a notable resource in the field of graph learning, particularly in the area of computer science research. This dataset forms a directed graph representing the citation network among all Computer Science papers on arXiv, as indexed by the Microsoft Academic Graph (MAG). Each node in this network corresponds to a paper, and directed edges indicate citations. The dataset's primary challenge is predicting the 40 subject areas of arXiv CS papers, such as cs.AI, cs.LG, and cs.OS. The task is structured as a 40-class classification problem.",
    "wikics": "The Wiki CS dataset is a comprehensive collection of Wikipedia entries, systematically categorized into ten distinct areas of computer science.",
}


class ZeroG(nn.Module):
    """ZeroG model with LoRA fine-tuning of SentenceBERT.

    Args:
        R: Number of neighborhood aggregation rounds (default: 10)
        if_norm: Whether to normalize embeddings (default: True)
        device: Torch device
    """

    def __init__(self, R=10, if_norm=True, device=None):
        super().__init__()
        self.R = R
        self.if_norm = if_norm
        self._device = device or torch.device("cpu")

        # Load SentenceBERT with LoRA
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/multi-qa-distilbert-cos-v1")
        textmodel = AutoModel.from_pretrained(
            "sentence-transformers/multi-qa-distilbert-cos-v1")

        lora_config = LoraConfig(
            task_type="SEQ_CLS",
            r=4,
            lora_alpha=16,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=0.1,
        )
        self.lora_model = LoraModel(textmodel, lora_config, "default")
        self.descriptions = DATASET_DESCRIPTIONS
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, data, dataset_name=None):
        """Training forward: compute loss on a subgraph.

        Args:
            data: PyG Data with .raw_text, .label_text, .y, .edge_index,
                  .dataset_name
            dataset_name: Override dataset name (if not in data)

        Returns:
            Cross-entropy loss (scalar)
        """
        ds_name = dataset_name or data.dataset_name
        device = self._device

        # Add virtual prompt node
        virtual_desc = self.descriptions[ds_name]
        all_texts = data.raw_text + [virtual_desc]

        # Encode all texts
        tokens = self.tokenizer(
            all_texts, max_length=256, return_tensors="pt",
            truncation=True, padding=True).to(device)
        node_embeds = self.lora_model(**tokens)[0][:, 0, :]

        # Encode labels
        tokens = self.tokenizer(
            data.label_text, max_length=256, return_tensors="pt",
            truncation=True, padding=True).to(device)
        label_embeds = self.lora_model(**tokens)[0][:, 0, :]

        if self.if_norm:
            node_embeds = (node_embeds - node_embeds.mean(0)) / node_embeds.std(0)
            label_embeds = (label_embeds - label_embeds.mean(0)) / label_embeds.std(0)

        # Build adjacency with virtual node
        num_nodes = data.y.shape[0] + 1
        virtual_idx = data.y.shape[0]

        # Bidirectional edges to virtual node (matches original for most datasets)
        new_edges = []
        for nid in range(num_nodes - 1):
            new_edges.append([nid, virtual_idx])
            new_edges.append([virtual_idx, nid])
        new_edge_index = torch.cat([
            data.edge_index.t(),
            torch.tensor(new_edges, dtype=torch.long).to(device)
        ], dim=0).t()

        # Normalize adjacency and propagate
        adj = self._normalize_adj(new_edge_index, num_nodes)
        for _ in range(self.R):
            node_embeds = torch.mm(adj, node_embeds)

        # Remove virtual node, compute logits
        node_embeds = node_embeds[:-1, :]
        logits = torch.mm(node_embeds, label_embeds.t())
        labels = data.y.long().to(device)
        if labels.dim() > 1:
            labels = labels.squeeze(1)

        return self.criteria(logits, labels)

    @torch.no_grad()
    def encode_graph(self, data, dataset_name):
        """Encode all nodes and labels of a full graph for inference.

        Args:
            data: PyG Data with .raw_texts, .label_name/.label_text,
                  .edge_index, .y
            dataset_name: Name for prompt node description

        Returns:
            logits: [N, C] similarity scores (cosine after propagation)
        """
        device = self._device

        # Encode node texts (batch to avoid OOM on large graphs)
        raw_texts = data.raw_texts
        desc = self.descriptions.get(dataset_name, f"A graph dataset called {dataset_name}.")
        all_texts = list(raw_texts) + [desc]

        node_embeds = self._batch_encode(all_texts)

        # Encode label texts
        label_texts = data.label_text if hasattr(data, "label_text") else data.label_name
        label_embeds = self._batch_encode(list(label_texts))

        if self.if_norm:
            node_embeds = (node_embeds - node_embeds.mean(0)) / (node_embeds.std(0) + 1e-8)
            label_embeds = (label_embeds - label_embeds.mean(0)) / (label_embeds.std(0) + 1e-8)

        # Build adjacency with virtual node
        num_nodes = data.y.shape[0] + 1
        virtual_idx = data.y.shape[0]
        new_edges = []
        for nid in range(num_nodes - 1):
            new_edges.append([nid, virtual_idx])
            new_edges.append([virtual_idx, nid])
        new_edge_index = torch.cat([
            data.edge_index.to(device).t(),
            torch.tensor(new_edges, dtype=torch.long, device=device)
        ], dim=0).t()

        adj = self._normalize_adj(new_edge_index, num_nodes)

        # Propagate R rounds
        node_embeds = node_embeds.to(device)
        for _ in range(self.R):
            node_embeds = torch.mm(adj, node_embeds)

        # Remove virtual node, normalize, compute similarity
        node_embeds = node_embeds[:-1, :]
        node_embeds = node_embeds / (node_embeds.norm(dim=-1, keepdim=True) + 1e-8)
        label_embeds = label_embeds.to(device)
        label_embeds = label_embeds / (label_embeds.norm(dim=-1, keepdim=True) + 1e-8)

        logits = torch.einsum("bn,cn->bc", node_embeds, label_embeds)
        return logits

    def _batch_encode(self, texts, batch_size=64):
        """Encode texts in batches to avoid OOM."""
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = self.tokenizer(
                batch, max_length=256, return_tensors="pt",
                truncation=True, padding=True).to(self._device)
            embeds = self.lora_model(**tokens)[0][:, 0, :]
            all_embeds.append(embeds.cpu())
        return torch.cat(all_embeds, dim=0)

    def _normalize_adj(self, edge_index, num_nodes):
        """Symmetric normalization: D^{-1/2} A D^{-1/2} with self-loops."""
        device = edge_index.device
        # Add self-loops
        self_loops = torch.stack([
            torch.arange(num_nodes, device=device),
            torch.arange(num_nodes, device=device)
        ], dim=0)
        edge_index = torch.cat([edge_index, self_loops], dim=1)

        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1], device=device),
            (num_nodes, num_nodes),
        )
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        D = torch.sparse_coo_tensor(
            torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1),
            deg_inv_sqrt,
            (num_nodes, num_nodes),
        )
        return torch.sparse.mm(D, torch.sparse.mm(adj, D))