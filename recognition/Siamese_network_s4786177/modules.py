"""
modules.py
Student name: Jay Thakkar
Student number: s4786177
Description: Implements a ResNet50-based feature extractor with a small MLP head.
Stage: ResNet components + TripletLoss (no classifier/wrapper yet)
"""

from __future__ import annotations  # use future annotations for forward refs in type hints
from typing import Optional

import torch                            # core tensor library
import torch.nn as nn                   # neural network modules
from torchvision import models          # pretrained ResNet50 (allowed to use from ed)
import torch.nn.functional as F         # functional API for activations, losses, etc.


# -------------------------- components --------------------------
class FeatureExtractor(nn.Module):
    """ResNet50 backbone with a small MLP projection head.

    This module extracts high-level features from input images using a
    ResNet50 backbone (with the final fully-connected layer removed), then
    passes the resulting 2048-d feature vector through a small MLP to
    produce a lower-dimensional embedding.

    Args:
        emb_dim: Dimension of the output embedding (default: 128).
        drop: Dropout probability used inside the MLP head.
        pretrained: If True, load ImageNet pretrained weights for ResNet50.
        l2_normalize: If True, L2-normalize output embeddings along dim=1.

    Returns:
        A module mapping an image tensor [B, C, H, W] to embeddings [B, emb_dim].
    """
    def __init__(
        self,
        emb_dim: int = 128,
        drop: float = 0.5,
        pretrained: bool = True,
        l2_normalize: bool = False,
    ) -> None:
        super().__init__()
        self.l2_normalize = l2_normalize

        # Load ResNet50; replace the classification head with Identity
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.resnet50(weights=weights)
        in_features = base.fc.in_features  # dimension of penultimate features
        base.fc = nn.Identity()            # strip ImageNet classifier
        self.base = base

        # Small MLP projection head: 2048 -> 512 -> 256 -> emb_dim
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, emb_dim),
        )

        # Kaiming initialization for linear layers in the MLP head
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for a batch of images.

        Args:
            x: Image tensor of shape [B, C, H, W].

        Returns:
            emb: Embedding tensor of shape [B, emb_dim]. If ``l2_normalize``
                 was set in the constructor the embeddings are L2-normalized
                 along dimension 1.
        """
        # Extract high-level features from the backbone (shape [B, 2048])
        feats = self.base(x)     # [B, 2048] for ResNet50

        # Map features to the embedding space
        emb = self.head(feats)   # [B, emb_dim]

        # Optionally L2-normalize embeddings (common for metric-learning setups)
        if self.l2_normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        return emb

class ClassifierHead(nn.Module):
    """Simple linear classification head operating on embeddings.

    This module maps embeddings to class logits. It is intentionally small
    (single linear layer) because classification is expected to be performed
    on top of learned embeddings rather than within a deep classifier here.

    Args:
        emb_dim: Dimensionality of input embeddings.
        num_classes: Number of output classes (default: 2).

    Forward:
        Input: emb tensor [B, emb_dim]
        Output: logits [B, num_classes]
    """
    def __init__(self, emb_dim: int = 128, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

        # Use Kaiming init on weights; small linear head so biases zeros are fine
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Return raw logits for each class.

        Note: No softmax applied here; return raw logits suitable for
        using with nn.CrossEntropyLoss.
        """
        return self.fc(emb)


class SiameseNet(nn.Module):
    """Convenience wrapper returning both embeddings and classification logits.

    This module composes a FeatureExtractor and a ClassifierHead. It is
    useful when training jointly for metric and classification losses or
    during evaluation when both embedding and logits are required.

    Args:
        emb_dim: Embedding dimensionality passed to FeatureExtractor.
        drop: Dropout probability for the FeatureExtractor head.
        pretrained: Whether to initialize the backbone with ImageNet weights.

    Forward:
        Input: image tensor [B, C, H, W]
        Output: tuple(embeddings [B, emb_dim], logits [B, num_classes])
    """
    def __init__(self, emb_dim: int = 128, drop: float = 0.6, pretrained: bool = True) -> None:
        super().__init__()
        # Feature extractor produces L2-normalized embeddings by default here
        self.fe  = FeatureExtractor(emb_dim=emb_dim, drop=drop, pretrained=pretrained, l2_normalize=True)
        self.clf = ClassifierHead(emb_dim=emb_dim, num_classes=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute embeddings and logits for a batch of images.

        Returns:
            emb: L2-normalized embeddings [B, emb_dim]
            logits: raw class logits [B, num_classes]
        """
        emb = self.fe(x)
        logits = self.clf(emb)
        return emb, logits


