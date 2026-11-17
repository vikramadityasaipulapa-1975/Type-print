"""Load and validate feature/sequence artifacts written by prepare_dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreparedDataset:
    frame: pd.DataFrame
    sequences: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    sequence_feature_names: tuple[str, ...]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def load_prepared_dataset(features_csv: str | Path, sequences_npz: str | Path) -> PreparedDataset:
    frame = pd.read_csv(features_csv)
    required = {"sample_id", "session_id", "user_id"}
    if not required.issubset(frame.columns):
        raise ValueError("feature CSV must contain sample_id, session_id, and user_id columns")
    if frame.empty or frame["user_id"].isna().any() or frame["session_id"].isna().any():
        raise ValueError("training requires non-empty samples with user_id labels and session_id values")

    with np.load(sequences_npz, allow_pickle=False) as data:
        sequences = data["sequences"].astype(np.float32)
        mask = data["mask"].astype(bool)
        lengths = data["lengths"].astype(np.int64)
        train_indices = data["train_indices"].astype(np.int64)
        validation_indices = data["validation_indices"].astype(np.int64)
        test_indices = data["test_indices"].astype(np.int64)
        sequence_sessions = data["session_ids"].astype(str)
        sequence_sample_ids = data["sample_ids"].astype(str)
        sequence_user_ids = data["user_ids"].astype(str)
        sequence_feature_names = tuple(data["sequence_feature_names"].astype(str).tolist())

    count = len(frame)
    if sequences.ndim != 3 or sequences.shape[0] != count:
        raise ValueError("sequence archive sample count does not match feature CSV")
    if mask.shape != sequences.shape[:2] or lengths.shape != (count,):
        raise ValueError("sequence masks and lengths do not match the sequence tensor")
    if not np.isfinite(sequences).all() or np.any(lengths < 1) or np.any(lengths > sequences.shape[1]):
        raise ValueError("prepared sequences must be finite and have valid lengths")
    expected_mask = np.arange(sequences.shape[1])[None, :] < lengths[:, None]
    if not np.array_equal(mask, expected_mask):
        raise ValueError("sequence masks must mark each valid prefix and mask all padding")
    if sequences.shape[2] != len(sequence_feature_names):
        raise ValueError("sequence feature names do not match the sequence tensor")
    if len(sequence_sessions) != count or not np.array_equal(frame["session_id"].astype(str).to_numpy(), sequence_sessions):
        raise ValueError("feature CSV and sequence archive do not refer to the same ordered samples")
    if not frame["sample_id"].isna().any() and not np.array_equal(frame["sample_id"].astype(str).to_numpy(), sequence_sample_ids):
        raise ValueError("sample IDs differ between feature CSV and sequence archive")
    if not np.array_equal(frame["user_id"].astype(str).to_numpy(), sequence_user_ids):
        raise ValueError("user labels differ between feature CSV and sequence archive")

    split_arrays = (train_indices, validation_indices, test_indices)
    all_indices = np.concatenate(split_arrays)
    if len(all_indices) != count or set(all_indices.tolist()) != set(range(count)):
        raise ValueError("prepared train/validation/test indices must partition all sample rows exactly once")
    if any(len(np.unique(indices)) != len(indices) for indices in split_arrays):
        raise ValueError("a prepared split contains duplicate sample indices")
    session_sets = [{sequence_sessions[index] for index in indices} for indices in split_arrays]
    if any(session_sets[left] & session_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("session leakage detected across prepared splits")
    if any(len(indices) == 0 for indices in split_arrays):
        raise ValueError("prepared train, validation, and test splits must all contain samples")
    return PreparedDataset(
        frame=frame,
        sequences=sequences,
        mask=mask,
        lengths=lengths,
        sequence_feature_names=sequence_feature_names,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )
