"""Prepare timing-only features and session-level splits from a JSONL file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ml.data.load_dataset import load_jsonl
from ml.features.keystroke_features import SEQUENCE_FEATURES, extract_sample, features_to_frame
from ml.preprocessing.pipeline import prepare_sequences, split_by_session


def prepare_dataset(input_path: str | Path, output_dir: str | Path, random_state: int = 42) -> tuple[Path, Path]:
    records = load_jsonl(input_path)
    if not records:
        raise ValueError("input JSONL contains no samples")
    results = [extract_sample(record) for record in records]
    split = split_by_session(results, random_state=random_state)
    features = features_to_frame(results)
    sequences, mask, lengths = prepare_sequences([result.sequence for result in results])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    feature_path = destination / "features.csv"
    sequence_path = destination / "sequences.npz"
    features.to_csv(feature_path, index=False)
    np.savez_compressed(
        sequence_path,
        sequences=sequences,
        sequence_feature_names=np.asarray(SEQUENCE_FEATURES, dtype=str),
        mask=mask,
        lengths=lengths,
        sample_ids=np.asarray([result.sample_id or f"sample-{index}" for index, result in enumerate(results)], dtype=str),
        session_ids=np.asarray([result.session_id for result in results], dtype=str),
        user_ids=np.asarray([result.user_id or "" for result in results], dtype=str),
        train_indices=split.train,
        validation_indices=split.validation,
        test_indices=split.test,
    )
    return feature_path, sequence_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    feature_path, sequence_path = prepare_dataset(args.input_jsonl, args.output_dir, args.seed)
    print(f"Features: {feature_path}")
    print(f"Padded sequences and session splits: {sequence_path}")


if __name__ == "__main__":
    main()
