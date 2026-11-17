"""Export encrypted enrollment sequences into the timing-only JSONL contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import select

from app.core.security import decrypt_json
from app.database.session import SessionLocal
from app.models.entities import TypingSession

FAMILY_CODE = {0: "KeyA", 1: "Digit1", 2: "Space", 3: "Period", 4: "Backspace", 5: "KeyA"}


def sequence_to_events(sequence: list[list[float]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_up = 0.0
    for index, point in enumerate(sequence):
        if len(point) != 8:
            raise ValueError("enrollment sequence has an unsupported feature width")
        dwell_ms, stored_flight = float(point[0]), float(point[1])
        family_index = max(range(6), key=lambda category: point[category + 2])
        down_ms = 0.0 if index == 0 else previous_up + stored_flight
        up_ms = down_ms + dwell_ms
        events.append({
            "id": index,
            "code": FAMILY_CODE[family_index],
            "downMs": down_ms,
            "upMs": up_ms,
            "dwellMs": dwell_ms,
            "flightMs": None if index == 0 else stored_flight,
        })
        previous_up = up_ms
    return events


def export_enrollment_samples(output_path: str | Path) -> int:
    db = SessionLocal()
    try:
        rows = db.scalars(select(TypingSession).where(TypingSession.session_type == "enrollment").order_by(TypingSession.user_id, TypingSession.session_timestamp)).all()
        records = []
        for row in rows:
            context = f"typing:{row.user_id}:{row.id}"
            payload = decrypt_json(row.sample_metadata, context)
            sequence = payload.get("sequence")
            if not isinstance(sequence, list) or not sequence:
                continue
            records.append({
                "sample_id": row.id,
                "session_id": row.collection_session_id or row.id,
                "user_id": f"subject-{row.user_id}",
                "events": sequence_to_events(sequence),
            })
    finally:
        db.close()
    if not records:
        raise ValueError("no encrypted enrollment sequences are available for export")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_suffix(destination.suffix + ".tmp")
    with temp_destination.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    temp_destination.replace(destination)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ml/data/enrollment-samples.jsonl"))
    args = parser.parse_args()
    count = export_enrollment_samples(args.output)
    print(f"Exported {count} encrypted enrollment samples to {args.output}")


if __name__ == "__main__":
    main()
