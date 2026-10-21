import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SESSIONS_PATH = PROJECT_ROOT / "data" / "processed" / "isaac_sessions.jsonl"
WINDOWS_PATH = PROJECT_ROOT / "data" / "processed" / "isaac_behavioral_windows.jsonl"
PILOT_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "behavioral_sample_v0.1.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10
RANDOM_SEED = 42

def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} on line {line_number}: {exc}"
                ) from exc
    return records

def write_jsonl(records, path):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")

def session_year(session):
    timestamp = session.get("start_time")
    if not timestamp:
        return "unknown"
    return datetime.fromisoformat(timestamp).year

def load_forced_train_session_ids():
    if not PILOT_SAMPLE_PATH.exists():
        return set()
    sample = load_jsonl(PILOT_SAMPLE_PATH)
    return {row["session_id"] for row in sample}

def split_sessions_by_year(sessions, forced_train_ids):
    rng = random.Random(RANDOM_SEED)
    by_year = defaultdict(list)

    for session in sessions:
        by_year[session_year(session)].append(session)

    train, validation, test = [], [], []

    for year in sorted(by_year, key=str):
        year_sessions = by_year[year]

        forced = [
            s for s in year_sessions
            if s["session_id"] in forced_train_ids
        ]
        available = [
            s for s in year_sessions
            if s["session_id"] not in forced_train_ids
        ]

        rng.shuffle(available)

        total_year = len(year_sessions)
        target_test = min(round(total_year * TEST_FRACTION), len(available))
        test_part = available[:target_test]

        remaining = available[target_test:]
        target_validation = min(
            round(total_year * VALIDATION_FRACTION),
            len(remaining),
        )
        validation_part = remaining[:target_validation]
        train_part = remaining[target_validation:]

        train.extend(forced)
        train.extend(train_part)
        validation.extend(validation_part)
        test.extend(test_part)

    key = lambda session: session.get("start_time", "")
    train.sort(key=key)
    validation.sort(key=key)
    test.sort(key=key)

    return train, validation, test

def split_windows(windows, split_session_ids):
    result = {"train": [], "validation": [], "test": []}

    for window in windows:
        sid = window["session_id"]
        if sid in split_session_ids["train"]:
            result["train"].append(window)
        elif sid in split_session_ids["validation"]:
            result["validation"].append(window)
        elif sid in split_session_ids["test"]:
            result["test"].append(window)
        else:
            raise RuntimeError(
                f"Window {window['window_id']} belongs to unassigned session {sid}."
            )

    return result

def assert_disjoint(train, validation, test):
    train_ids = {s["session_id"] for s in train}
    val_ids = {s["session_id"] for s in validation}
    test_ids = {s["session_id"] for s in test}

    if train_ids & val_ids:
        raise RuntimeError("Train/validation session leakage detected.")
    if train_ids & test_ids:
        raise RuntimeError("Train/test session leakage detected.")
    if val_ids & test_ids:
        raise RuntimeError("Validation/test session leakage detected.")

def year_counts(records):
    return dict(
        sorted(
            Counter(session_year(record) for record in records).items(),
            key=lambda item: str(item[0]),
        )
    )

def main():
    sessions = load_jsonl(SESSIONS_PATH)
    windows = load_jsonl(WINDOWS_PATH)
    forced_train_ids = load_forced_train_session_ids()

    train, validation, test = split_sessions_by_year(
        sessions,
        forced_train_ids,
    )

    assert_disjoint(train, validation, test)

    ids = {
        "train": {s["session_id"] for s in train},
        "validation": {s["session_id"] for s in validation},
        "test": {s["session_id"] for s in test},
    }

    split_window_records = split_windows(windows, ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(train, OUTPUT_DIR / "train_sessions.jsonl")
    write_jsonl(validation, OUTPUT_DIR / "validation_sessions.jsonl")
    write_jsonl(test, OUTPUT_DIR / "test_sessions.jsonl")

    write_jsonl(
        split_window_records["train"],
        OUTPUT_DIR / "train_behavioral_windows.jsonl",
    )
    write_jsonl(
        split_window_records["validation"],
        OUTPUT_DIR / "validation_behavioral_windows.jsonl",
    )
    write_jsonl(
        split_window_records["test"],
        OUTPUT_DIR / "test_behavioral_windows.jsonl",
    )

    manifest = {
        "random_seed": RANDOM_SEED,
        "strategy": (
            "whole-session split, stratified by year; sessions used in "
            "behavioral_sample_v0.1 forced into train"
        ),
        "fractions_requested": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "forced_train_session_count": len(forced_train_ids),
        "sessions": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "windows": {
            split: len(rows)
            for split, rows in split_window_records.items()
        },
        "years": {
            "train": year_counts(train),
            "validation": year_counts(validation),
            "test": year_counts(test),
        },
    }

    with (OUTPUT_DIR / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print()
    print(f"Total sessions:                 {len(sessions):,}")
    print(f"Pilot sessions forced to train: {len(forced_train_ids):,}")
    print()
    print(f"Train sessions:                 {len(train):,}")
    print(f"Validation sessions:            {len(validation):,}")
    print(f"Test sessions:                  {len(test):,}")
    print()
    print(f"Train windows:                  {len(split_window_records['train']):,}")
    print(f"Validation windows:             {len(split_window_records['validation']):,}")
    print(f"Test windows:                   {len(split_window_records['test']):,}")
    print()
    print("Session overlap checks:         PASSED")
    print(f"Output directory:               {OUTPUT_DIR}")
    print()
    print(
        "IMPORTANT: rebuild the RAG index using ONLY "
        "train_behavioral_windows.jsonl before evaluating on test."
    )

if __name__ == "__main__":
    main()
