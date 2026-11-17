"""Configurable LSTM classifier for variable-length keystroke sequences."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class KeystrokeLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1 or num_classes < 2:
            raise ValueError("input_size, hidden_size, num_layers must be positive and num_classes must be at least 2")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_classes = num_classes
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def encode(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 3 or sequences.shape[2] != self.input_size:
            raise ValueError("sequences must have shape (batch, time, input_size)")
        if lengths.ndim != 1 or lengths.shape[0] != sequences.shape[0]:
            raise ValueError("lengths must contain one value per sequence")
        lengths_cpu = lengths.detach().to(device="cpu", dtype=torch.int64)
        if torch.any(lengths_cpu < 1) or torch.any(lengths_cpu > sequences.shape[1]):
            raise ValueError("sequence lengths must be between 1 and the padded sequence length")
        packed = pack_padded_sequence(sequences, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return hidden[-1]

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedding = self.dropout(self.encode(sequences, lengths))
        return self.classifier(embedding)
