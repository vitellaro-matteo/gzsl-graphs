"""
ICIS: Image-free Classifier Injection with Semantics.

Christensen et al., ICCV 2023. https://github.com/ExplainableML/ImageFreeZSL

Adapted from computer vision to graph data (thesis Chapter 5):
  - ResNet features -> GraphSAGE / MLP features
  - Visual attributes -> SentenceBERT CSD embeddings
  - CNN classifier weights -> MLP base classifier weights

Architecture:
  - Two autoencoders: AE_attribute (semantic space) and AE_weight (classifier weight space)
  - Wrapped in JOINT_AUTOENCODER that shares a latent space
  - Training: reconstruct within and across spaces (4 losses)
  - Inference: encode unseen class attributes -> decode to classifier weights
  - Inject predicted weights into extended classifier -> GZSL evaluation

This file contains model architectures only. Training in scripts/train_icis.py.
"""

import torch
import torch.nn as nn


# ============================================================
# Base classifiers
# ============================================================

class LinearClassifier(nn.Module):
    """Simple linear classifier (softmax layer)."""
    def __init__(self, input_dim, num_classes, bias=True):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias)

    def forward(self, x):
        return self.fc(x)


class MLPClassifier(nn.Module):
    """Multi-layer perceptron classifier for graph features."""
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)
        # Expose final layer as .fc for weight extraction compatibility
        self.fc = layers[-1]

    def forward(self, x):
        return self.net(x)


# ============================================================
# Autoencoders
# ============================================================

class Autoencoder(nn.Module):
    """Single autoencoder with configurable depth.

    Args:
        input_dim: Input dimension
        embed_dim: Latent embedding dimension
        output_dim: Output dimension (default: same as input_dim)
        num_layers: 2, 3, or 4 layer architecture
    """
    def __init__(self, input_dim, embed_dim, output_dim=None, num_layers=3):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        if num_layers == 2:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(inplace=True),
            )
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, output_dim),
            )
        elif num_layers == 3:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(inplace=True),
            )
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, output_dim),
            )
        elif num_layers == 4:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
            )
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, output_dim),
            )
        else:
            raise ValueError(f"num_layers must be 2, 3, or 4, got {num_layers}")

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))


class JointAutoencoder(nn.Module):
    """Joint autoencoder with shared latent space (ICIS core).

    Two autoencoders share a latent space Z:
      - AE_attribute: A -> Z -> A  (semantic descriptions)
      - AE_weight: W -> Z -> W    (classifier weights)

    Cross-reconstruction enables:
      - A -> Z -> W  (predict weights from attributes, used at inference)
      - W -> Z -> A  (regularization)

    Args:
        ae_attribute: Autoencoder for semantic attributes
        ae_weight: Autoencoder for classifier weights
    """
    def __init__(self, ae_attribute, ae_weight):
        super().__init__()
        self.ae_attribute = ae_attribute
        self.ae_weight = ae_weight

    def forward(self, x):
        """Forward pass with cross-reconstruction.

        Args:
            x: tuple (attributes, weights)

        Returns:
            att_from_att, att_from_weight, weight_from_weight, weight_from_att,
            latent_att, latent_weight
        """
        att_in, weight_in = x
        latent_att = self.ae_attribute.encode(att_in)
        latent_weight = self.ae_weight.encode(weight_in)

        att_from_att = self.ae_attribute.decode(latent_att)
        att_from_weight = self.ae_attribute.decode(latent_weight)
        weight_from_weight = self.ae_weight.decode(latent_weight)
        weight_from_att = self.ae_weight.decode(latent_att)

        return (att_from_att, att_from_weight, weight_from_weight,
                weight_from_att, latent_att, latent_weight)

    def predict(self, attributes):
        """Predict classifier weights from semantic attributes.

        This is the core ICIS inference: A -> Z -> W.

        Args:
            attributes: [N, attr_dim] semantic descriptions

        Returns:
            predicted_weights: [N, weight_dim] (weight + bias concatenated)
        """
        latent = self.ae_attribute.encode(attributes)
        return self.ae_weight.decode(latent)


def weights_init(m):
    """Initialize weights (from original ICIS code)."""
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)