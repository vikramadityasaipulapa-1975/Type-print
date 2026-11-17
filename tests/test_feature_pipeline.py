"""Unit tests use synthetic event fixtures; they are not biometric samples."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ml.data.load_dataset import load_jsonl
from ml.data.prepare_dataset import prepare_dataset
from ml.data.export_enrollment import sequence_to_events
from ml.training.calibrate_verifier import calibrate_verifier
from ml.biometrics.calibration import fit_distance_calibrator
from ml.biometrics.profile import create_profiles
from ml.evaluation.classification import evaluate_classifier
from ml.models.random_forest import build_random_forest
from ml.models.lstm import KeystrokeLSTM
from ml.inference.authentication import AuthenticationEngine
from ml.inference.lstm_encoder import LSTMEncoder
from ml.training.train_lstm import train_lstm
from ml.training.train_siamese import train_siamese
from ml.models.siamese import ContrastiveLoss, SiameseKeystrokeNetwork
from ml.training.siamese_pairs import make_balanced_pairs
from ml.training.train_random_forest import train_random_forest
from ml.features.keystroke_features import SampleValidationError, extract_sample
from ml.preprocessing.pipeline import SequenceRobustScaler, prepare_sequences, split_by_session


def event(index: int, code: str, down: float, up: float, flight: float | None) -> dict[str, object]:
    return {"id": index, "code": code, "downMs": down, "upMs": up, "dwellMs": up - down, "flightMs": flight}


def sample(session: str = "session-a") -> dict[str, object]:
    return {
        "sample_id": "fixture",
        "session_id": session,
        "user_id": "synthetic-subject",
        "events": [
            event(0, "KeyA", 0, 80, None),
            event(1, "Space", 100, 170, 20),
            event(2, "KeyB", 180, 250, 10),
            event(3, "Backspace", 300, 360, 50),
        ],
    }


class FeaturePipelineTests(unittest.TestCase):
    def test_extracts_stats_transition_and_correction_features(self) -> None:
        result = extract_sample(sample())
        self.assertEqual(result.sequence.shape, (4, 8))
        self.assertAlmostEqual(result.features["dwell_ms_mean"], 70.0)
        self.assertEqual(result.features["backspace_count"], 1.0)
        self.assertEqual(result.features["transition_alpha_space"], 1 / 3)
        self.assertEqual(result.accepted_event_count, 4)
        self.assertEqual(result.rejected_event_count, 0)

    def test_filters_duplicate_and_invalid_events(self) -> None:
        record = sample()
        events = record["events"]
        events.extend([events[1], {"code": "KeyZ", "downMs": 400, "upMs": 99, "dwellMs": 1}])
        result = extract_sample(record)
        self.assertEqual(result.accepted_event_count, 4)
        self.assertEqual(result.rejected_event_count, 2)

    def test_requires_session_id_and_sufficient_events(self) -> None:
        with self.assertRaises(SampleValidationError):
            extract_sample({"events": sample()["events"]})
        with self.assertRaises(SampleValidationError):
            extract_sample({"session_id": "short", "events": [event(0, "KeyA", 0, 10, None)]})

    def test_rejects_character_text_instead_of_physical_key_code(self) -> None:
        record = sample()
        record["events"][0]["code"] = "a"
        with self.assertRaises(SampleValidationError):
            extract_sample(record)

    def test_pads_variable_sequences_and_returns_boolean_mask(self) -> None:
        one = np.array([[10.0, np.nan, 1.0], [20.0, 2.0, 1.0]], dtype=np.float32)
        two = np.array([[30.0, 3.0, 0.0]], dtype=np.float32)
        padded, mask, lengths = prepare_sequences([one, two], max_length=3)
        self.assertEqual(padded.shape, (2, 3, 3))
        self.assertEqual(mask.tolist(), [[True, True, False], [True, False, False]])
        self.assertEqual(lengths.tolist(), [2, 1])
        self.assertTrue(np.isfinite(padded).all())
        self.assertEqual(float(padded[0, 2, 0]), 0.0)

    def test_scaler_fit_and_transform_valid_timesteps_only(self) -> None:
        values, mask, _ = prepare_sequences([np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0]])], max_length=2)
        scaler = SequenceRobustScaler().fit(values[:1], mask[:1])
        transformed = scaler.transform(values, mask)
        self.assertEqual(transformed.shape, values.shape)
        self.assertTrue(np.all(transformed[~mask] == 0))

    def test_split_keeps_sessions_disjoint(self) -> None:
        records = [{"session_id": f"s-{index}"} for index in range(20)]
        split = split_by_session(records, random_state=7)
        session_sets = [{records[index]["session_id"] for index in part} for part in (split.train, split.validation, split.test)]
        self.assertFalse(session_sets[0] & session_sets[1])
        self.assertFalse(session_sets[0] & session_sets[2])
        self.assertFalse(session_sets[1] & session_sets[2])
        self.assertEqual(sum(map(len, (split.train, split.validation, split.test))), len(records))

    def test_dataset_loader_rejects_raw_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "samples.jsonl"
            path.write_text(json.dumps({"session_id": "s", "events": [{"code": "KeyA", "key": "a"}]}) + "\n", encoding="utf-8")
            with self.assertRaises(SampleValidationError):
                load_jsonl(path)

    def test_pandas_frame_keeps_labels_separate_and_schema_stable(self) -> None:
        result = extract_sample(sample())
        from ml.features.keystroke_features import features_to_frame
        frame = features_to_frame([result])
        self.assertIn("session_id", frame.columns)
        self.assertIn("user_id", frame.columns)
        self.assertNotIn("text", frame.columns)
        self.assertEqual(frame.loc[0, "session_id"], "session-a")

    def test_encrypted_sequence_export_contract_reconstructs_timing_only_events(self) -> None:
        original = extract_sample(sample())
        events = sequence_to_events(original.sequence.tolist())
        exported = extract_sample({"session_id": "export-session", "events": events})
        self.assertEqual(len(events), original.accepted_event_count)
        self.assertAlmostEqual(exported.features["dwell_ms_mean"], original.features["dwell_ms_mean"])
        self.assertEqual(events[0]["code"], "KeyA")
        self.assertNotIn("text", events[0])

    def test_dataset_builder_writes_features_and_session_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "samples.jsonl"
            records = []
            for index in range(6):
                record = sample(f"session-{index}")
                record["sample_id"] = f"sample-{index}"
                records.append(record)
            source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            features_path, sequences_path = prepare_dataset(source, root / "out")
            self.assertTrue(features_path.exists())
            self.assertTrue(sequences_path.exists())
            with np.load(sequences_path, allow_pickle=False) as data:
                self.assertEqual(data["sequences"].shape, (6, 4, 8))
                self.assertEqual(data["sequence_feature_names"].tolist()[0:2], ["dwell_ms", "flight_ms"])
                self.assertEqual(len(data["train_indices"]) + len(data["validation_indices"]) + len(data["test_indices"]), 6)

    def test_random_forest_and_lstm_baselines_train_and_write_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "synthetic-test-only.jsonl"
            records = []
            for user_index in range(4):
                base_dwell = 45 + 35 * user_index
                for session_index in range(10):
                    dwell = base_dwell + (session_index % 3)
                    flight = 15 + 12 * user_index + session_index % 2
                    events = [
                        event(0, "KeyA", 0, dwell, None),
                        event(1, "Space", dwell + flight, 2 * dwell + flight, flight),
                        event(2, "KeyB", 2 * dwell + flight + flight + 5, 3 * dwell + 2 * flight + 5, flight + 5),
                        event(3, "Period", 3 * dwell + 3 * flight + 10, 4 * dwell + 3 * flight + 10, flight + 5),
                    ]
                    records.append({"sample_id": f"{user_index}-{session_index}", "session_id": f"session-{user_index}-{session_index}", "user_id": f"subject-{user_index}", "events": events})
            source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            features_path, sequences_path = prepare_dataset(source, root / "prepared", random_state=17)
            model_path = root / "models" / "rf.joblib"
            metrics = train_random_forest(features_path, sequences_path, model_path, n_estimators=20, random_state=17)
            self.assertEqual(metrics["class_count"], 4)
            self.assertTrue(model_path.exists())
            self.assertTrue(model_path.with_suffix(".joblib.metrics.json").exists())
            self.assertEqual(metrics["test"]["sample_count"], metrics["split_sample_counts"]["test"])
            self.assertIn("classification_roc_auc_ovr_weighted", metrics["test"])
            self.assertIn("FAR", " ".join(metrics["limitations"]))
            old_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                checkpoint_path = root / "models" / "lstm.pt"
                lstm_metrics = train_lstm(
                    features_path,
                    sequences_path,
                    checkpoint_path,
                    hidden_size=8,
                    num_layers=1,
                    dropout=0,
                    epochs=3,
                    batch_size=8,
                    patience=2,
                    seed=17,
                    device_name="cpu",
                )
                self.assertTrue(checkpoint_path.exists())
                self.assertTrue(checkpoint_path.with_suffix(".pt.metrics.json").exists())
                self.assertEqual(lstm_metrics["class_count"], 4)
                self.assertGreaterEqual(lstm_metrics["epochs_completed"], 1)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["model_config"]["num_classes"], 4)
                encoder = LSTMEncoder.from_checkpoint(checkpoint_path)
                embedding = encoder.encode_sequence(extract_sample(sample()).sequence)
                self.assertEqual(embedding.shape, (8,))
                self.assertTrue(np.isfinite(embedding).all())
                verification_path = root / "models" / "verification.metrics.json"
                verification = calibrate_verifier(
                    features_path,
                    sequences_path,
                    checkpoint_path,
                    verification_path,
                    calibrator_path=root / "models" / "distance_calibrator.joblib",
                    min_enrollment_samples=5,
                )
                self.assertTrue(verification_path.exists())
                self.assertTrue((root / "models" / "distance_calibrator.joblib").exists())
                self.assertIn("far", verification["test_verification"])
                self.assertIn("eer", verification["test_verification"])
            finally:
                torch.set_num_threads(old_threads)

    def test_siamese_pairs_model_loss_and_training(self) -> None:
        pairs = make_balanced_pairs(
            np.asarray(["a", "a", "b", "b"]), np.arange(4), pairs_per_sample=1, seed=3
        )
        self.assertEqual(set(pairs[:, 2].tolist()), {0, 1})
        model = SiameseKeystrokeNetwork(input_size=8, hidden_size=6, num_layers=1, dropout=0, embedding_size=4)
        first = torch.randn(3, 5, 8)
        second = torch.randn(3, 5, 8)
        lengths = torch.tensor([3, 4, 5])
        left, right = model(first, lengths, second, lengths)
        loss = ContrastiveLoss()(left, right, torch.tensor([1, 0, 1]))
        loss.backward()
        self.assertEqual(left.shape, (3, 4))
        self.assertTrue(torch.isfinite(loss))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "synthetic-siamese-test-only.jsonl"
            records = []
            for user_index in range(4):
                base_dwell = 45 + 35 * user_index
                for session_index in range(10):
                    dwell, flight = base_dwell + session_index % 3, 15 + 12 * user_index
                    events = [
                        event(0, "KeyA", 0, dwell, None),
                        event(1, "Space", dwell + flight, 2 * dwell + flight, flight),
                        event(2, "KeyB", 2 * dwell + flight + flight + 5, 3 * dwell + 2 * flight + 5, flight + 5),
                        event(3, "Period", 3 * dwell + 3 * flight + 10, 4 * dwell + 3 * flight + 10, flight + 5),
                    ]
                    records.append({"sample_id": f"{user_index}-{session_index}", "session_id": f"siamese-{user_index}-{session_index}", "user_id": f"subject-{user_index}", "events": events})
            source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            features_path, sequences_path = prepare_dataset(source, root / "prepared", random_state=17)
            old_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                checkpoint_path = root / "models" / "siamese.pt"
                metrics = train_siamese(features_path, sequences_path, checkpoint_path, hidden_size=8, num_layers=1, embedding_size=4, dropout=0, epochs=2, batch_size=8, patience=2, seed=17, device_name="cpu")
                self.assertTrue(checkpoint_path.exists())
                self.assertTrue(checkpoint_path.with_suffix(".pt.metrics.json").exists())
                self.assertIn("eer", metrics["test"])
            finally:
                torch.set_num_threads(old_threads)

    def test_closed_set_evaluation_includes_roc_auc_when_all_classes_are_present(self) -> None:
        train_x = np.asarray([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
        train_y = np.asarray(["a", "a", "a", "b", "b", "b"])
        classifier = build_random_forest(n_estimators=20, random_state=3).fit(train_x, train_y)
        metrics = evaluate_classifier(classifier, np.asarray([[0.15], [1.15]]), np.asarray(["a", "b"]), np.asarray(["a", "b"]))
        self.assertIsNotNone(metrics["classification_roc_auc_ovr_weighted"])
        self.assertEqual(len(metrics["confusion_matrix"]), 2)

    def test_lstm_ignores_padded_timesteps(self) -> None:
        model = KeystrokeLSTM(input_size=8, hidden_size=10, num_layers=1, dropout=0, num_classes=3).eval()
        values = torch.randn(2, 5, 8)
        lengths = torch.tensor([2, 4])
        changed_padding = values.clone()
        changed_padding[0, 2:] = 1000
        changed_padding[1, 4:] = -1000
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.inference_mode():
                self.assertTrue(torch.allclose(model(values, lengths), model(changed_padding, lengths), atol=1e-6))
        finally:
            torch.set_num_threads(old_threads)

    def test_enrollment_calibration_and_verification_decisions(self) -> None:
        alice_train = np.asarray([[-0.08, 0.02], [-0.03, -0.04], [0.01, 0.02], [0.04, -0.02], [0.08, 0.01]])
        bob_train = np.asarray([[2.92, 3.02], [2.97, 2.96], [3.01, 3.02], [3.04, 2.98], [3.08, 3.01]])
        profiles = create_profiles(
            np.vstack([alice_train, bob_train]),
            np.asarray(["alice"] * 5 + ["bob"] * 5),
            model_version="lstm-test-v1",
            min_samples=5,
        )
        calibration_embeddings = np.asarray([[-0.02, 0.01], [0.02, -0.02], [2.98, 3.01], [3.02, 2.99]])
        calibration_subjects = np.asarray(["alice", "alice", "bob", "bob"])
        calibrator, calibration_metrics = fit_distance_calibrator(profiles, calibration_embeddings, calibration_subjects, "lstm-test-v1")
        self.assertIn("far", calibration_metrics)
        class ArrayEncoder:
            model_version = "lstm-test-v1"
            def encode_sequence(self, sequence: np.ndarray) -> np.ndarray:
                return np.asarray(sequence[0], dtype=np.float32)
        engine = AuthenticationEngine(ArrayEncoder(), profiles, calibrator, minimum_sequence_length=2)
        self.assertEqual(engine.verify("unknown", np.ones((2, 2))).status, "unknown_user")
        self.assertEqual(engine.verify("alice", np.ones((1, 2))).status, "insufficient_data")
        self.assertEqual(engine.verify("alice", np.asarray([[-0.01, 0.0], [-0.01, 0.0]])).status, "authenticated")
        self.assertEqual(engine.verify("alice", np.asarray([[3.0, 3.0], [3.0, 3.0]])).status, "rejected")
        near_score = calibrator.score(profiles["alice"].distance(np.asarray([-0.01, 0.0])))
        uncertain_engine = AuthenticationEngine(ArrayEncoder(), profiles, calibrator, threshold=near_score, minimum_sequence_length=2)
        self.assertEqual(uncertain_engine.verify("alice", np.asarray([[-0.01, 0.0], [-0.01, 0.0]])).status, "low_confidence")

    def test_calibration_fails_closed_when_distances_do_not_separate(self) -> None:
        values = np.zeros((10, 2), dtype=np.float32)
        labels = np.asarray(["alice"] * 5 + ["bob"] * 5)
        profiles = create_profiles(values, labels, "same-embeddings-v1", min_samples=5)
        with self.assertRaisesRegex(ValueError, "does not separate"):
            fit_distance_calibrator(
                profiles,
                np.zeros((4, 2)),
                np.asarray(["alice", "alice", "bob", "bob"]),
                "same-embeddings-v1",
            )


if __name__ == "__main__":
    unittest.main()
