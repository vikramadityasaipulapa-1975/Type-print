"""Shared-weight sequence encoder and contrastive objective for verification."""

from __future__ import annotations

import torch
from torch import nn

from ml.models.lstm import KeystrokeLSTM


class SiameseKeystrokeNetwork(nn.Module):
    """Encode both samples with the same LSTM and return their embeddings."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        embedding_size: int = 32,
    ) -> None:
        super().__init__()
        if embedding_size < 1:
            raise ValueError("embedding_size must be positive")
        self.encoder = KeystrokeLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=2,
        )
        self.projection = nn.Sequential(nn.Linear(hidden_size, embedding_size), nn.LayerNorm(embedding_size))
        self.embedding_size = embedding_size

    def encode(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.projection(self.encoder.encode(sequences, lengths)), p=2, dim=1)

    def forward(
        self,
        first: torch.Tensor,
        first_lengths: torch.Tensor,
        second: torch.Tensor,
        second_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(first, first_lengths), self.encode(second, second_lengths)


class ContrastiveLoss(nn.Module):
    """Labels are 1 for a genuine pair and 0 for an impostor pair."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError("margin must be positive")
        self.margin = margin

    def forward(self, first: torch.Tensor, second: torch.Tensor, same_identity: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape or first.ndim != 2:
            raise ValueError("embeddings must be matching two-dimensional tensors")
        labels = same_identity.to(device=first.device, dtype=first.dtype).reshape(-1)
        if labels.shape[0] != first.shape[0] or torch.any((labels != 0) & (labels != 1)):
            raise ValueError("same_identity must contain one binary label per pair")
        distance = torch.linalg.vector_norm(first - second, ord=2, dim=1)
        genuine_loss = labels * distance.square()
        impostor_loss = (1 - labels) * torch.clamp(self.margin - distance, min=0).square()
        return (genuine_loss + impostor_loss).mean()
