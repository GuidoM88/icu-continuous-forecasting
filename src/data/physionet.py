"""PhysioNet/CinC Challenge 2012 parser."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

GENERAL_DESCRIPTORS = ["Age", "Gender", "Height", "ICUType", "Weight"]


@dataclass
class PatientRecord:
    record_id: str
    static: Dict[str, float]
    times: List[float]
    values: Dict[str, List[Tuple[float, float]]]
    label: Optional[int] = None


def _time_to_hours(value: str) -> float:
    hh, mm = value.split(":")
    return int(hh) + int(mm) / 60.0


def parse_record_file(path: Path) -> PatientRecord:
    static: Dict[str, float] = {}
    series: Dict[str, List[Tuple[float, float]]] = {}
    record_id = path.stem

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if not row or len(row) < 3:
                continue

            time_str = row[0].strip()
            param = row[1].strip()
            value_str = row[2].strip()

            if not time_str or not param or not value_str:
                continue

            try:
                value = float(value_str)
                t = _time_to_hours(time_str)
            except (ValueError, TypeError):
                continue

            if param == "RecordID":
                record_id = str(int(value))
                continue

            if param in GENERAL_DESCRIPTORS:
                if value != -1:
                    static[param] = value
                continue

            series.setdefault(param, []).append((t, value))

    times = sorted({t for pts in series.values() for t, _ in pts})
    return PatientRecord(
        record_id=record_id,
        static=static,
        times=times,
        values=series,
    )


def load_outcomes(path: Path) -> Dict[str, int]:
    outcomes = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            outcomes[str(int(row["RecordID"]))] = int(row["In-hospital_death"])
    return outcomes


def load_split(
    raw_dir: Path,
    split: str = "set-a",
    outcomes_file: Optional[str] = None,
) -> List[PatientRecord]:
    split_dir = raw_dir / split
    files = sorted(split_dir.glob("*.txt"))

    if not files:
        raise FileNotFoundError(f"No .txt records found under {split_dir}")

    if outcomes_file is None and split == "set-a":
        outcomes_file = "Outcomes-a.txt"

    outcomes = load_outcomes(raw_dir / outcomes_file) if outcomes_file else {}

    records = []
    for path in files:
        rec = parse_record_file(path)
        rec.label = outcomes.get(rec.record_id)
        records.append(rec)

    return records
