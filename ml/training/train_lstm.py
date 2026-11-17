"""Train an LSTM classifier on prepared sequences with early stopping."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from ml.data.prepared import load_prepared_dataset
from ml.models.lstm import KeystrokeLSTM
from ml.preprocessing.pipeline import SequenceRobustScaler
from ml.training.sequence_dataset import TypingSequenceDataset


def _evaluate(model: KeystrokeLSTM, loader: DataLoader, classes: np.ndarray, class_weights: torch.Tensor, device: torch.device) -> tuple[float, dict[str, Any]]:
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model.eval()
    losses: list[float] = []
    true_values: list[int] = []
    predicted_values: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for sequences, lengths, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            logits = model(sequences, lengths)
            losses.append(float(criterion(logits, labels).item()) * len(labels))
            true_values.extend(labels.cpu().numpy().tolist())
            predicted_values.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
    y_true = np.asarray(true_values, dtype=np.int64)
    y_pred = np.asarray(predicted_values, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    result: dict[str, Any] = {
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=np.arange(len(classes)), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=np.arange(len(classes)), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(len(classes)), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(classes))).tolist(),
        "confusion_matrix_labels": [str(value) for value in classes],
        "classification_report": classification_report(
            y_true, y_pred, labels=np.arange(len(classes)), target_names=[str(value) for value in classes], output_dict=True, zero_division=0
        ),
    }
    if len(classes) > 1 and set(y_true.tolist()) == set(range(len(classes))):
        if len(classes) == 2:
            result["classification_roc_auc_ovr_weighted"] = float(roc_auc_score((y_true == 1).astype(np.int8), probs[:, 1]))
        else:
            result["classification_roc_auc_ovr_weighted"] = float(
                roc_auc_score(y_true, probs, labels=np.arange(len(classes)), multi_class="ovr", average="weighted")
            )
    else:
        result["classification_roc_auc_ovr_weighted"] = None
        result["roc_auc_note"] = "ROC-AUC omitted because this evaluation split does not contain every training class."
    return sum(losses) / len(y_true), result


def train_lstm(
    features_csv: str | Path,
    sequences_npz: str | Path,
    checkpoint_path: str | Path,
    *,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 10,
    seed: int = 42,
    device_name: str | None = None,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, patience, and learning_rate must be positive")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dataset = load_prepared_dataset(features_csv, sequences_npz)
    feature_columns = [column for column in dataset.frame.columns if column not in {"sample_id", "session_id", "user_id"}]
    if not feature_columns:
        raise ValueError("feature CSV contains no statistical model features")
    try:
        features = dataset.frame[feature_columns].to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("statistical features must be numeric") from exc
    if np.isinf(features).any():
        raise ValueError("statistical features cannot contain positive or negative infinity")

    labels = dataset.frame["user_id"].astype(str).to_numpy()
    classes = np.unique(labels[dataset.train_indices])
    if len(classes) < 2:
        raise ValueError("LSTM identity classification requires at least two users in the training split")
    unknown = set(labels[np.concatenate((dataset.validation_indices, dataset.test_indices))]) - set(classes)
    if unknown:
        raise ValueError(f"validation/test contains users missing from training: {sorted(unknown)}")
    class_to_index = {name: index for index, name in enumerate(classes.tolist())}
    encoded_labels = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)

    train_idx = dataset.train_indices
    validation_idx = dataset.validation_indices
    test_idx = dataset.test_indices
    scaler = SequenceRobustScaler().fit(dataset.sequences[train_idx], dataset.mask[train_idx])
    sequences = scaler.transform(dataset.sequences, dataset.mask)
    train_data = TypingSequenceDataset(sequences[train_idx], dataset.lengths[train_idx], encoded_labels[train_idx])
    validation_data = TypingSequenceDataset(sequences[validation_idx], dataset.lengths[validation_idx], encoded_labels[validation_idx])
    test_data = TypingSequenceDataset(sequences[test_idx], dataset.lengths[test_idx], encoded_labels[test_idx])

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = KeystrokeLSTM(
        input_size=dataset.sequences.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=len(classes),
    ).to(device)
    train_class_counts = np.bincount(encoded_labels[train_idx], minlength=len(classes)).astype(np.float32)
    class_weights = torch.tensor(len(train_idx) / (len(classes) * train_class_counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_sequences, batch_lengths, batch_labels in train_loader:
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_sequences, batch_lengths)
            loss = criterion(logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_labels)
        train_loss = total_loss / len(train_data)
        validation_loss, _ = _evaluate(model, validation_loader, classes, class_weights, device)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    validation_loss, validation_metrics = _evaluate(model, validation_loader, classes, class_weights, device)
    test_loss, test_metrics = _evaluate(model, test_loader, classes, class_weights, device)
    metrics: dict[str, Any] = {
        "model_name": "lstm_identity_classifier",
        "model_version": "0.1.0",
        "class_count": int(len(classes)),
        "classes": [str(value) for value in classes],
        "split_sample_counts": {"train": int(len(train_idx)), "validation": int(len(validation_idx)), "test": int(len(test_idx))},
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "validation_loss": validation_loss,
        "test_loss": test_loss,
        "validation": validation_metrics,
        "test": test_metrics,
        "limitations": [
            "Metrics require real consented user samples and are only as representative as the collection sessions.",
            "Closed-set classification logits and probabilities are not calibrated biometric match scores or FAR/FRR/EER.",
            "This model cannot recognize a user class absent from its training split.",
        ],
    }
    checkpoint = {
        "model_name": "lstm_identity_classifier",
        "model_version": "0.1.0",
        "model_state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "model_config": {"input_size": int(dataset.sequences.shape[2]), "hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout, "num_classes": int(len(classes))},
        "classes": [str(value) for value in classes],
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
    metrics_path = destination.with_suffix(destination.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_csv", type=Path)
    parser.add_argument("sequences_npz", type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("models/keystroke_lstm.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or omit to auto-select")
    args = parser.parse_args()
    metrics = train_lstm(
        args.features_csv,
        args.sequences_npz,
        args.checkpoint_path,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
