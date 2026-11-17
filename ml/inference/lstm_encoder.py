"""Load an LSTM checkpoint and generate scaled sequence embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ml.models.lstm import KeystrokeLSTM


class LSTMEncoder:
    def __init__(
        self,
        model: KeystrokeLSTM,
        center: np.ndarray,
        scale: np.ndarray,
        model_version: str,
        sequence_feature_names: list[str],
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.model_version = model_version
        self.sequence_feature_names = tuple(sequence_feature_names)
        if self.center.shape != (model.input_size,) or self.scale.shape != (model.input_size,) or np.any(self.scale <= 0):
            raise ValueError("checkpoint scaler does not match the LSTM input dimension")

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "LSTMEncoder":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        required = {"model_state_dict", "model_config", "scaler_center", "scaler_scale", "sequence_feature_names"}
        if not required.issubset(checkpoint):
            raise ValueError("LSTM checkpoint is missing inference metadata")
        config = checkpoint["model_config"]
        model = KeystrokeLSTM(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(
            model=model,
            center=np.asarray(checkpoint["scaler_center"], dtype=np.float32),
            scale=np.asarray(checkpoint["scaler_scale"], dtype=np.float32),
            model_version=str(checkpoint.get("model_version", checkpoint.get("metrics", {}).get("model_version", "unknown"))),
            sequence_feature_names=checkpoint["sequence_feature_names"],
            device=device,
        )

    def encode_sequence(self, sequence: np.ndarray) -> np.ndarray:
        values = np.asarray(sequence, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != self.model.input_size:
            raise ValueError("sequence must have shape (timesteps, checkpoint input_size)")
        if not np.isfinite(values).all():
            raise ValueError("sequence values must be finite")
        scaled = (values - self.center) / self.scale
        tensor = torch.from_numpy(scaled).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(values)], dtype=torch.int64)
        with torch.inference_mode():
            embedding = self.model.encode(tensor, lengths)
        return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
