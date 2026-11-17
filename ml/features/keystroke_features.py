"""Validation and feature extraction for timing-only keystroke samples.

The raw sample contract contains session identifiers and key categories/timing
only. Do not add captured passage text or arbitrary KeyboardEvent.key values.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import re
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

MAX_DWELL_MS = 2_500.0
MIN_FLIGHT_MS = -2_500.0
MAX_FLIGHT_MS = 10_000.0
KEY_CODE_PATTERN = re.compile(r"^(Key[A-Z]|Digit[0-9]|Numpad[0-9]|Space|Backspace|Minus|Equal|BracketLeft|BracketRight|Backslash|Semicolon|Quote|Comma|Period|Slash|Backquote)$")
KEY_FAMILIES = ("alpha", "digit", "space", "punct", "correction", "other")
KEY_FAMILY_INDEX = {name: index for index, name in enumerate(KEY_FAMILIES)}
SEQUENCE_FEATURES = ("dwell_ms", "flight_ms", *(f"key_family_{family}" for family in KEY_FAMILIES))


class SampleValidationError(ValueError):
    """Raised when a sample cannot yield a trustworthy timing sequence."""


@dataclass(frozen=True)
class FeatureResult:
    sample_id: str | None
    session_id: str
    user_id: str | None
    features: dict[str, float]
    sequence: np.ndarray
    accepted_event_count: int
    rejected_event_count: int


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def key_family(code: Any) -> str:
    if not isinstance(code, str):
        return "other"
    if code.startswith("Key") and len(code) == 4:
        return "alpha"
    if code.startswith("Digit") or code.startswith("Numpad"):
        return "digit"
    if code == "Space":
        return "space"
    if code == "Backspace":
        return "correction"
    if code in {"Minus", "Equal", "BracketLeft", "BracketRight", "Backslash", "Semicolon", "Quote", "Comma", "Period", "Slash", "Backquote"}:
        return "punct"
    return "other"


def _clip_iqr(values: Sequence[float]) -> list[float]:
    """Winsorize sample-local extremes; physical bounds are filtered first."""
    if len(values) < 4:
        return list(values)
    q1, q3 = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.75])
    spread = float(q3 - q1)
    if spread == 0:
        return list(values)
    lower, upper = float(q1 - 1.5 * spread), float(q3 + 1.5 * spread)
    return [float(np.clip(value, lower, upper)) for value in values]


def _stats(name: str, values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {f"{name}_{stat}": 0.0 for stat in ("mean", "median", "std", "min", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{name}_mean": float(np.mean(array)),
        f"{name}_median": float(np.median(array)),
        f"{name}_std": float(np.std(array)),
        f"{name}_min": float(np.min(array)),
        f"{name}_max": float(np.max(array)),
    }


def _parse_events(events: Any) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise SampleValidationError("events must be a sequence of key timing records")

    parsed: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    rejected = 0
    for source_order, raw in enumerate(events):
        if not isinstance(raw, Mapping):
            rejected += 1
            continue
        if "key" in raw or "text" in raw or "typed_text" in raw or "content" in raw:
            raise SampleValidationError("raw key text and typed content are prohibited")
        code = raw.get("code")
        if not isinstance(code, str) or not KEY_CODE_PATTERN.fullmatch(code):
            raise SampleValidationError("code must be an allowed physical key category, not key text")
        if len(code) > 32:
            rejected += 1
            continue
        down = _number(raw.get("downMs"))
        up = _number(raw.get("upMs"))
        dwell = _number(raw.get("dwellMs"))
        flight = _number(raw.get("flightMs"))
        if dwell is None and down is not None and up is not None:
            dwell = up - down
        if dwell is None or not (0 <= dwell <= MAX_DWELL_MS):
            rejected += 1
            continue
        if (down is not None and down < 0) or (up is not None and up < 0):
            rejected += 1
            continue
        if down is not None and up is not None and up < down:
            rejected += 1
            continue
        if flight is not None and not (MIN_FLIGHT_MS <= flight <= MAX_FLIGHT_MS):
            rejected += 1
            continue
        signature = (code, down, up, dwell, flight)
        if signature in seen:
            rejected += 1
            continue
        seen.add(signature)
        parsed.append({"code": code, "down": down, "up": up, "dwell": dwell, "flight": flight, "order": source_order})

    parsed.sort(key=lambda item: (item["down"] is None, item["down"] if item["down"] is not None else item["order"], item["order"]))
    previous_up: float | None = None
    for event in parsed:
        if event["flight"] is None and event["down"] is not None and previous_up is not None:
            event["flight"] = event["down"] - previous_up
        if event["flight"] is not None and not (MIN_FLIGHT_MS <= event["flight"] <= MAX_FLIGHT_MS):
            event["flight"] = None
        if event["up"] is not None:
            previous_up = event["up"]
    return parsed, rejected


def extract_sample(sample: Mapping[str, Any]) -> FeatureResult:
    """Extract robust statistical and sequential features from one session sample."""
    if not isinstance(sample, Mapping):
        raise SampleValidationError("sample must be a mapping")
    if "text" in sample or "typed_text" in sample or "content" in sample:
        raise SampleValidationError("raw typed content is prohibited")
    session_id = sample.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise SampleValidationError("session_id is required for leakage-safe splitting")
    events, rejected = _parse_events(sample.get("events"))
    if len(events) < 2:
        raise SampleValidationError("at least two valid key events are required")

    dwell = _clip_iqr([event["dwell"] for event in events])
    flight_indices = [index for index, event in enumerate(events) if event["flight"] is not None]
    flights = _clip_iqr([events[index]["flight"] for index in flight_indices])
    for index, clipped in zip(flight_indices, flights):
        events[index]["flight"] = clipped
    downs = [event["down"] for event in events if event["down"] is not None]
    keydown_intervals = [right - left for left, right in zip(downs, downs[1:]) if right >= left]
    keydown_intervals = _clip_iqr(keydown_intervals)

    features: dict[str, float] = {}
    features.update(_stats("dwell_ms", dwell))
    features.update(_stats("flight_ms", flights))
    features.update(_stats("keydown_interval_ms", keydown_intervals))
    durations = [event["up"] for event in events if event["up"] is not None]
    if downs and durations:
        elapsed = max(durations) - min(downs)
    else:
        elapsed = sum(dwell)
    backspaces = sum(key_family(event["code"]) == "correction" for event in events)
    char_count = max(0, sum(key_family(event["code"]) != "correction" for event in events) - backspaces)
    features["duration_ms"] = float(max(0.0, elapsed))
    features["characters_per_minute"] = float(char_count * 60_000 / elapsed) if elapsed > 0 else 0.0
    features["typing_consistency"] = float(1 / (1 + (np.std(keydown_intervals) / np.mean(keydown_intervals)))) if keydown_intervals and np.mean(keydown_intervals) > 0 else 0.0
    features["keypress_count"] = float(len(events))
    features["backspace_count"] = float(backspaces)
    features["correction_ratio"] = float(backspaces / len(events))

    family_path = [key_family(event["code"]) for event in events]
    transition_counts = {f"transition_{left}_{right}": 0.0 for left in KEY_FAMILIES for right in KEY_FAMILIES}
    for left, right in zip(family_path, family_path[1:]):
        transition_counts[f"transition_{left}_{right}"] += 1.0
    transition_total = max(1, len(family_path) - 1)
    features.update({name: count / transition_total for name, count in transition_counts.items()})

    flight_values = [event["flight"] for event in events if event["flight"] is not None]
    fallback_flight = float(median(flight_values)) if flight_values else 0.0
    sequence_values = []
    for index, (event, dwell_value) in enumerate(zip(events, dwell)):
        flight_value = event["flight"]
        if flight_value is None:
            flight_value = fallback_flight
        family_index = KEY_FAMILY_INDEX[family_path[index]]
        family_one_hot = [1.0 if category_index == family_index else 0.0 for category_index in range(len(KEY_FAMILIES))]
        sequence_values.append([dwell_value, float(flight_value), *family_one_hot])

    return FeatureResult(
        sample_id=sample.get("sample_id") if isinstance(sample.get("sample_id"), str) else None,
        session_id=session_id,
        user_id=sample.get("user_id") if isinstance(sample.get("user_id"), str) else None,
        features=features,
        sequence=np.asarray(sequence_values, dtype=np.float32),
        accepted_event_count=len(events),
        rejected_event_count=rejected,
    )


def features_to_frame(results: Sequence[FeatureResult]) -> pd.DataFrame:
    """Build a stable pandas frame with labels/IDs separate from model features."""
    if not results:
        return pd.DataFrame(columns=["sample_id", "session_id", "user_id"])
    feature_names = sorted(results[0].features)
    if any(sorted(result.features) != feature_names for result in results):
        raise ValueError("feature schemas differ between samples")
    rows = [
        {"sample_id": result.sample_id, "session_id": result.session_id, "user_id": result.user_id, **result.features}
        for result in results
    ]
    return pd.DataFrame(rows, columns=["sample_id", "session_id", "user_id", *feature_names])
