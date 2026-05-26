"""Model registry for graph ZSL methods."""

from .dgpn import DGPN, compute_dgpn_loss
from .dbigcn import DBiGCN, compute_dbigcn_loss, build_class_adjacency, build_node_adjacency
from .zerog import ZeroG
from .icis import JointAutoencoder, Autoencoder, LinearClassifier, MLPClassifier

MODEL_REGISTRY = {
    "dgpn": DGPN,
    "dbigcn": DBiGCN,
    "zerog": ZeroG,
    "icis": JointAutoencoder,
}


def get_model(name, **kwargs):
    """Factory: instantiate a model by name."""
    assert name in MODEL_REGISTRY, f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
    return MODEL_REGISTRY[name](**kwargs)