"""
Parser for the raw PhysioNet/CinC Challenge 2012 dataset
("Predicting Mortality of ICU Patients").

Expected raw layout, exactly as distributed by PhysioNet (see
`data/download_physionet2012.sh`):

    data/raw/set-a/132539.txt
    data/raw/set-a/132540.txt
    ...
    data/raw/Outcomes-a.txt

Each per-patient .txt file is a 3-column CSV: Time,Parameter,Value
Time is given as "HH:MM" elapsed since ICU admission (e.g. "35:19" = 35h19m
after admission). The first rows, all stamped 00:00, are six general
descriptors: RecordID, Age, Gender, Height, ICUType, Weight -- everything
else is a repeated-measure clinical variable (up to ~37 of them, not all
present for every patient).

Outcomes-a.txt is a CSV with columns:
RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death

A value of -1 for a general descriptor means "not recorded". Only set-a
ships outcomes to participants -- sets B and C are the (label-free) test
sets from the original Challenge and are not used here.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

GENERAL_DESCRIPTORS = ["Age", "Gender", "Height", "ICUType", "Weight"]


@dataclass
class PatientRecord:
    record_id: str
    static: Dict[str, float]                 # e.g. {"Age": 65.0, "Gender": 1.0, ...}
    times: List[float]                        # observation times in hours, sorted-unique
    values: Dict[str, List[Tuple[float, float]]]  # variable_name -> [(time, value), ...]
    label: Optional[int] = None               # in-hospital death (0/1), if known


def _time_to_hours(t: str) -> float:
    hh, mm = t.split(":")
    return int(hh) + int(mm) / 60.0


def parse_record_file(path: Path) -> PatientRecord:
    static: Dict[str, float] = {}
    series: Dict[str, List[Tuple[float, float]]] = {}
    record_id = path.stem

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header: "Time,Parameter,Value"
        for row in reader:
            if not row or len(row) < 3:
                continue
            time_str = row[0].strip()
            param = row[1].strip()
            value_str = row[2].strip()

            # Some released files may contain malformed/blank CSV rows.
            # An empty Parameter is not a physiological variable and must
            # never enter the model vocabulary.
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
                if value != -1:  # -1 == "not recorded", per the Challenge spec
                    static[param] = value
                continue

            series.setdefault(param, []).append((t, value))

    times = sorted({t for pts in series.values() for t, _ in pts})
    return PatientRecord(record_id=record_id, static=static, times=times, values=series)


def load_outcomes(path: Path) -> Dict[str, int]:
    """Returns {record_id: in_hospital_death (0/1)}."""
    outcomes = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            outcomes[str(int(row["RecordID"]))] = int(row["In-hospital_death"])
    return outcomes


def load_split(
    raw_dir: Path, split: str = "set-a", outcomes_file: Optional[str] = None
) -> List[PatientRecord]:
    """Parse every record in a split directory.

    Competition-safe default:
      - set-a automatically uses Outcomes-a.txt;
      - set-b / set-c remain unlabeled unless an outcomes file is explicitly
        supplied for retrospective scoring.
    """
    split_dir = raw_dir / split
    files = sorted(split_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No .txt records found under {split_dir}. Did you run "
            f"data/download_physionet2012.sh ?"
        )
    if outcomes_file is None and split == "set-a":
        outcomes_file = "Outcomes-a.txt"
    outcomes = load_outcomes(raw_dir / outcomes_file) if outcomes_file else {}

    records = []
    for f in files:
        rec = parse_record_file(f)
        rec.label = outcomes.get(rec.record_id)
        records.append(rec)
    return records
