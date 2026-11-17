"""Leakage-conscious scaling, sequence padding, masking, and data splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class SessionSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _get_sequence(value: Any) -> np.ndarray:
    sequence = value.sequence if hasattr(value, "sequence") else value
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("each sequence must have shape (timesteps, features)")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("sequences cannot be empty")
    return array


def prepare_sequences(
    sequences: Iterable[Any],
    max_length: int | None = None,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return padded float32 sequences, valid-timestep mask, and original lengths.

    Missing and non-finite values are imputed with the finite median for that
    sample/channel (0 if a channel has no finite values). Long samples retain
    their most recent `max_length` events. Padding is added at the end.
    """
    arrays = [_get_sequence(item) for item in sequences]
    if not arrays:
        raise ValueError("at least one sequence is required")
    feature_count = arrays[0].shape[1]
    if any(array.shape[1] != feature_count for array in arrays):
        raise ValueError("all sequences must have the same feature count")
    if max_length is not None and max_length < 1:
        raise ValueError("max_length must be positive")

    lengths = np.asarray([min(len(array), max_length) if max_length else len(array) for array in arrays], dtype=np.int64)
    padded_length = max_length or max(len(array) for array in arrays)
    batch = np.full((len(arrays), padded_length, feature_count), pad_value, dtype=np.float32)
    mask = np.zeros((len(arrays), padded_length), dtype=bool)
    for row, array in enumerate(arrays):
        array = array[-padded_length:].copy()
        for column in range(feature_count):
            values = array[:, column]
            finite = np.isfinite(values)
            fill = float(np.median(values[finite])) if finite.any() else 0.0
            values[~finite] = fill
        count = len(array)
        batch[row, :count] = array
        mask[row, :count] = True
    return batch, mask, lengths


class SequenceRobustScaler:
    """Robust scale features, fit only on the training sequences supplied."""

    def __init__(self) -> None:
        self._scaler = RobustScaler()
        self._fitted = False

    def fit(self, sequences: np.ndarray, mask: np.ndarray) -> "SequenceRobustScaler":
        sequences = np.asarray(sequences, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if sequences.ndim != 3 or mask.shape != sequences.shape[:2]:
            raise ValueError("expected sequences (samples, time, features) and matching mask")
        valid = sequences[mask]
        if len(valid) == 0:
            raise ValueError("cannot fit scaler without valid timesteps")
        self._scaler.fit(valid)
        self._fitted = True
        return self

    @property
    def center_(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit the scaler before accessing center_")
        return self._scaler.center_

    @property
    def scale_(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit the scaler before accessing scale_")
        return self._scaler.scale_

    def transform(self, sequences: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit the scaler on training data before transform")
        values = np.asarray(sequences, dtype=np.float32)
        valid_mask = np.asarray(mask, dtype=bool)
        if values.ndim != 3 or valid_mask.shape != values.shape[:2]:
            raise ValueError("expected sequences (samples, time, features) and matching mask")
        transformed = np.zeros_like(values)
        transformed[valid_mask] = self._scaler.transform(values[valid_mask]).astype(np.float32)
        return transformed


def split_by_session(
    samples: Sequence[Any],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
) -> SessionSplit:
    """Split sample indices by session ID so one session cannot leak across sets."""
    fractions = np.asarray([train_fraction, validation_fraction, test_fraction], dtype=np.float64)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("split fractions must be positive and sum to 1")
    groups = []
    for sample in samples:
        session_id = sample.get("session_id") if isinstance(sample, dict) else getattr(sample, "session_id", None)
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("every sample must have a session_id")
        groups.append(session_id)
    groups_array = np.asarray(groups, dtype=object)
    unique_groups = np.unique(groups_array)
    if len(unique_groups) < 3:
        raise ValueError("at least three distinct collection sessions are required for train/validation/test split")

    all_indices = np.arange(len(samples))
    first = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    train_validation, test = next(first.split(all_indices, groups=groups_array))
    relative_validation = validation_fraction / (train_fraction + validation_fraction)
    second = GroupShuffleSplit(n_splits=1, test_size=relative_validation, random_state=random_state + 1)
    train_local, validation_local = next(second.split(train_validation, groups=groups_array[train_validation]))
    train = train_validation[train_local]
    validation = train_validation[validation_local]
    return SessionSplit(train=train, validation=validation, test=test)
