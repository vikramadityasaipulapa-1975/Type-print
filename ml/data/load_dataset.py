"""Read timing-only sample records from JSON Lines without retaining raw text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.features.keystroke_features import SampleValidationError


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records with session_id/events; raw text fields are rejected."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SampleValidationError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise SampleValidationError(f"line {line_number} must contain a JSON object")
            allowed_fields = {"sample_id", "session_id", "user_id", "events"}
            if set(record) - allowed_fields:
                raise SampleValidationError(f"line {line_number} contains unsupported fields")
            if not isinstance(record.get("session_id"), str) or not isinstance(record.get("events"), list):
                raise SampleValidationError(f"line {line_number} requires session_id and events")
            allowed_event_fields = {"id", "code", "downMs", "upMs", "dwellMs", "flightMs"}
            if any(not isinstance(event, dict) or set(event) - allowed_event_fields for event in record["events"]):
                raise SampleValidationError(f"line {line_number} contains unsupported event fields")
            records.append(record)
    return records
