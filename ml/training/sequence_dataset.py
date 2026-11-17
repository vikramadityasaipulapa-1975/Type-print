"""PyTorch Dataset for padded keystroke sequences."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TypingSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, lengths: np.ndarray, labels: np.ndarray) -> None:
        sequences = np.asarray(sequences, dtype=np.float32)
        lengths = np.asarray(lengths, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64)
        if sequences.ndim != 3 or lengths.shape != (len(sequences),) or labels.shape != (len(sequences),):
            raise ValueError("sequences, lengths, and labels have incompatible shapes")
        if np.any(lengths < 1) or np.any(lengths > sequences.shape[1]):
            raise ValueError("sequence lengths must be within padded tensor bounds")
        self.sequences = torch.from_numpy(sequences)
        self.lengths = torch.from_numpy(lengths)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.lengths[index], self.labels[index]
