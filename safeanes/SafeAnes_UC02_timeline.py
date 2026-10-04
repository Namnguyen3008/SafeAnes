"""Deterministic draft timeline construction from UC01 records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from safeanes.SafeAnes_UC01_data_contracts import ClinicalEvent, Observation, SignalSegment


@dataclass(frozen=True)
class TimelineItem:
    occurred_at: datetime
    kind: str
    record_id: str
    record: Observation | SignalSegment | ClinicalEvent


@dataclass(frozen=True)
class DraftTimeline:
    subject_id: str
    items: tuple[TimelineItem, ...]
    status: str = "DRAFT"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.status != "DRAFT":
            raise ValueError("This subsystem can create drafts only; signing is not implemented")


def build_draft_timeline(
    observations: Iterable[Observation] = (),
    events: Iterable[ClinicalEvent] = (),
    signal_segments: Iterable[SignalSegment] = (),
    *,
    subject_id: str | None = None,
) -> DraftTimeline:
    """Merge validated records into a stable timeline without inferring events."""
    observation_records = tuple(observations)
    event_records = tuple(events)
    signal_records = tuple(signal_segments)

    subjects = {
        record.subject_id
        for record in (*observation_records, *event_records, *signal_records)
    }
    if subject_id is not None:
        if not subject_id.strip():
            raise ValueError("subject_id must be non-empty")
        if subjects and subjects != {subject_id}:
            raise ValueError("All timeline records must belong to the same subject")
        resolved_subject = subject_id
    elif len(subjects) == 1:
        resolved_subject = next(iter(subjects))
    elif not subjects:
        raise ValueError("An empty draft timeline requires an explicit subject_id")
    else:
        raise ValueError("All timeline records must belong to the same subject")

    entries = [
        TimelineItem(record.observed_at, "observation", record.observation_id, record)
        for record in observation_records
    ]
    entries.extend(
        TimelineItem(record.started_at, "event", record.event_id, record)
        for record in event_records
    )
    entries.extend(
        TimelineItem(record.started_at, "signal_segment", record.segment_id, record)
        for record in signal_records
    )

    identifiers: set[tuple[str, str]] = set()
    for item in entries:
        key = (item.kind, item.record_id)
        if key in identifiers:
            raise ValueError(f"Duplicate {item.kind} id in timeline: {item.record_id}")
        identifiers.add(key)

    entries.sort(
        key=lambda item: (
            item.occurred_at,
            {"event": 0, "observation": 1, "signal_segment": 2}[item.kind],
            item.record_id,
        )
    )
    return DraftTimeline(subject_id=resolved_subject, items=tuple(entries))
