"""Session-safe pair construction for Siamese training and evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequencePairDataset(Dataset):
    def __init__(self, sequences: np.ndarray, lengths: np.ndarray, labels: np.ndarray, pairs: np.ndarray) -> None:
        sequences = np.asarray(sequences, dtype=np.float32)
        lengths = np.asarray(lengths, dtype=np.int64)
        labels = np.asarray(labels).astype(str)
        pairs = np.asarray(pairs, dtype=np.int64)
        if sequences.ndim != 3 or lengths.shape != (len(sequences),) or labels.shape != (len(sequences),):
            raise ValueError("sequences, lengths, and labels have incompatible shapes")
        if pairs.ndim != 2 or pairs.shape[1] != 3:
            raise ValueError("pairs must have columns (first_index, second_index, same_identity)")
        if len(pairs) and (np.any(pairs[:, :2] < 0) or np.any(pairs[:, :2] >= len(sequences))):
            raise ValueError("pair indices are outside the sequence array")
        if len(pairs) and np.any((pairs[:, 2] != 0) & (pairs[:, 2] != 1)):
            raise ValueError("pair labels must be binary")
        if np.any(lengths < 1) or np.any(lengths > sequences.shape[1]):
            raise ValueError("sequence lengths must be within padded tensor bounds")
        self.sequences = torch.from_numpy(sequences)
        self.lengths = torch.from_numpy(lengths)
        self.labels = torch.from_numpy(pairs[:, 2].astype(np.float32))
        self.first_indices = torch.from_numpy(pairs[:, 0])
        self.second_indices = torch.from_numpy(pairs[:, 1])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        first = self.first_indices[index]
        second = self.second_indices[index]
        return self.sequences[first], self.lengths[first], self.sequences[second], self.lengths[second], self.labels[index]


def make_balanced_pairs(labels: np.ndarray, indices: np.ndarray, *, pairs_per_sample: int = 2, seed: int = 42) -> np.ndarray:
    """Return balanced genuine/impostor pairs using only the requested split."""
    labels = np.asarray(labels).astype(str)
    indices = np.asarray(indices, dtype=np.int64)
    if pairs_per_sample < 1:
        raise ValueError("pairs_per_sample must be positive")
    if indices.ndim != 1 or len(indices) < 2 or np.any(indices < 0) or np.any(indices >= len(labels)):
        raise ValueError("indices must select at least two valid samples")
    by_user = {user: indices[labels[indices] == user] for user in np.unique(labels[indices])}
    by_user = {user: members for user, members in by_user.items() if len(members) >= 2}
    if len(by_user) < 2:
        raise ValueError("pair construction needs two users with at least two samples each")
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int, int]] = []
    for anchor in indices:
        same_pool = by_user.get(labels[anchor])
        if same_pool is None:
            continue
        positives = same_pool[same_pool != anchor]
        negatives = indices[labels[indices] != labels[anchor]]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        for _ in range(pairs_per_sample):
            pairs.append((int(anchor), int(rng.choice(positives)), 1))
            pairs.append((int(anchor), int(rng.choice(negatives)), 0))
    rng.shuffle(pairs)
    return np.asarray(pairs, dtype=np.int64)
