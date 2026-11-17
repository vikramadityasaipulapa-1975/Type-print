"""Train an optional Siamese verifier using session-disjoint sample pairs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader

from ml.data.prepared import load_prepared_dataset
from ml.models.siamese import ContrastiveLoss, SiameseKeystrokeNetwork
from ml.preprocessing.pipeline import SequenceRobustScaler
from ml.training.siamese_pairs import SequencePairDataset, make_balanced_pairs


def _evaluate(model: SiameseKeystrokeNetwork, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    losses: list[float] = []
    distances: list[float] = []
    targets: list[int] = []
    with torch.inference_mode():
        for first, first_len, second, second_len, same in loader:
            first, second = first.to(device), second.to(device)
            first_embedding, second_embedding = model(first, first_len, second, second_len)
            losses.append(float(criterion(first_embedding, second_embedding, same.to(device)).item()) * len(same))
            distances.extend(torch.linalg.vector_norm(first_embedding - second_embedding, dim=1).cpu().numpy().tolist())
            targets.extend(same.to(torch.int64).numpy().tolist())
    y = np.asarray(targets, dtype=np.int8)
    d = np.asarray(distances, dtype=np.float64)
    if len(np.unique(y)) != 2:
        raise ValueError("evaluation pairs must include genuine and impostor examples")
    # Similarity is negative distance, so positive ROC labels denote genuine pairs.
    fpr, tpr, thresholds = roc_curve(y, -d, pos_label=1)
    eer_index = int(np.argmin(np.abs(fpr - (1 - tpr))))
    distance_threshold = float(-thresholds[eer_index])
    predicted = (d <= distance_threshold).astype(np.int8)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    return float(sum(losses) / len(y)), {
        "pair_count": int(len(y)),
        "roc_auc": float(roc_auc_score(y, -d)),
        "eer": float((fpr[eer_index] + 1 - tpr[eer_index]) / 2),
        "distance_threshold_at_eer": distance_threshold,
        "far": float(matrix[0, 1] / max(1, matrix[0].sum())),
        "frr": float(matrix[1, 0] / max(1, matrix[1].sum())),
        "confusion_matrix": matrix.tolist(),
    }


def train_siamese(
    features_csv: str | Path,
    sequences_npz: str | Path,
    checkpoint_path: str | Path,
    *,
    hidden_size: int = 64,
    num_layers: int = 2,
    embedding_size: int = 32,
    dropout: float = 0.2,
    margin: float = 1.0,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    learning_rate: float = 1e-3,
    pairs_per_sample: int = 2,
    seed: int = 42,
    device_name: str | None = None,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, patience, and learning_rate must be positive")
    torch.manual_seed(seed)
    dataset = load_prepared_dataset(features_csv, sequences_npz)
    labels = dataset.frame["user_id"].astype(str).to_numpy()
    train_idx, validation_idx, test_idx = dataset.train_indices, dataset.validation_indices, dataset.test_indices
    scaler = SequenceRobustScaler().fit(dataset.sequences[train_idx], dataset.mask[train_idx])
    sequences = scaler.transform(dataset.sequences, dataset.mask)
    pair_sets = {
        "train": make_balanced_pairs(labels, train_idx, pairs_per_sample=pairs_per_sample, seed=seed),
        "validation": make_balanced_pairs(labels, validation_idx, pairs_per_sample=pairs_per_sample, seed=seed + 1),
        "test": make_balanced_pairs(labels, test_idx, pairs_per_sample=pairs_per_sample, seed=seed + 2),
    }
    loaders = {
        name: DataLoader(
            SequencePairDataset(sequences, dataset.lengths, labels, pairs),
            batch_size=batch_size,
            shuffle=name == "train",
            generator=torch.Generator().manual_seed(seed) if name == "train" else None,
        )
        for name, pairs in pair_sets.items()
    }
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SiameseKeystrokeNetwork(
        input_size=dataset.sequences.shape[2], hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout, embedding_size=embedding_size,
    ).to(device)
    criterion = ContrastiveLoss(margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for first, first_len, second, second_len, same in loaders["train"]:
            first, second = first.to(device), second.to(device)
            optimizer.zero_grad(set_to_none=True)
            first_embedding, second_embedding = model(first, first_len, second, second_len)
            loss = criterion(first_embedding, second_embedding, same.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(same)
            count += len(same)
        validation_loss, _ = _evaluate(model, loaders["validation"], criterion, device)
        history.append({"epoch": float(epoch), "train_loss": total / count, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss, best_state, best_epoch, stale = validation_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    _, validation_metrics = _evaluate(model, loaders["validation"], criterion, device)
    _, test_metrics = _evaluate(model, loaders["test"], criterion, device)
    metrics = {
        "model_name": "siamese_keystroke_verifier",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "validation": validation_metrics,
        "test": test_metrics,
        "limitations": ["Thresholds must be calibrated on representative consented enrollment and impostor data before deployment."],
    }
    checkpoint = {
        "model_name": "siamese_keystroke_verifier",
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": {"input_size": int(dataset.sequences.shape[2]), "hidden_size": hidden_size, "num_layers": num_layers, "embedding_size": embedding_size, "dropout": dropout, "margin": margin},
        "sequence_feature_names": list(dataset.sequence_feature_names),
        "scaler_center": scaler.center_.astype(np.float32).tolist(),
        "scaler_scale": scaler.scale_.astype(np.float32).tolist(),
        "metrics": metrics,
        "history": history,
        "seed": seed,
    }
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    destination.with_suffix(destination.suffix + ".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_csv", type=Path)
    parser.add_argument("sequences_npz", type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("models/siamese.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    print(json.dumps(train_siamese(args.features_csv, args.sequences_npz, args.checkpoint_path, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, seed=args.seed, device_name=args.device), indent=2))


if __name__ == "__main__":
    main()
