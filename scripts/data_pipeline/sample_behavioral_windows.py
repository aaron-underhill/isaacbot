import json
import random
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_PATH = Path("isaac_behavioral_windows.jsonl")
OUTPUT_PATH = Path("behavioral_sample_v0.1.jsonl")

SAMPLE_SIZE = 200
MIN_TURNS = 4
CHRONOLOGICAL_BINS = 20
RANDOM_SEED = 42


# =========================================================
# IO
# =========================================================

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
                    f"Invalid JSON in {path} on line "
                    f"{line_number}: {exc}"
                ) from exc

    return records


def write_jsonl(records, path):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


# =========================================================
# SAMPLING
# =========================================================

def chronological_bins(records, bin_count):
    """
    Split already-sorted records into roughly equal chronological bins.
    """
    if not records:
        return []

    bins = []

    total = len(records)

    for i in range(bin_count):
        start = round(i * total / bin_count)
        end = round((i + 1) * total / bin_count)

        chunk = records[start:end]

        if chunk:
            bins.append(chunk)

    return bins


def sample_unique_sessions_from_bin(records, target, rng, used_sessions):
    """
    Prefer windows from sessions not already represented in the sample.
    """
    candidates = records[:]
    rng.shuffle(candidates)

    chosen = []

    # First pass: unused sessions only.
    for record in candidates:
        if len(chosen) >= target:
            break

        session_id = record["session_id"]

        if session_id in used_sessions:
            continue

        chosen.append(record)
        used_sessions.add(session_id)

    # Second pass: if the bin did not have enough unique sessions,
    # allow additional windows from already-used sessions.
    if len(chosen) < target:
        chosen_ids = {
            record["window_id"]
            for record in chosen
        }

        for record in candidates:
            if len(chosen) >= target:
                break

            if record["window_id"] in chosen_ids:
                continue

            chosen.append(record)

    return chosen


def main():
    rng = random.Random(RANDOM_SEED)

    windows = load_jsonl(INPUT_PATH)

    eligible = [
        window
        for window in windows
        if window.get("window_turn_count", 0) >= MIN_TURNS
    ]

    eligible.sort(
        key=lambda window: window["start_time"]
    )

    bins = chronological_bins(
        eligible,
        CHRONOLOGICAL_BINS,
    )

    if not bins:
        raise RuntimeError("No eligible windows found.")

    base_target = SAMPLE_SIZE // len(bins)
    remainder = SAMPLE_SIZE % len(bins)

    sample = []
    used_sessions = set()

    for index, bin_records in enumerate(bins):
        target = base_target + (
            1 if index < remainder else 0
        )

        chosen = sample_unique_sessions_from_bin(
            bin_records,
            target,
            rng,
            used_sessions,
        )

        sample.extend(chosen)

    # If we are still short because of small bins, top up from all
    # remaining eligible windows.
    if len(sample) < SAMPLE_SIZE:
        selected_window_ids = {
            record["window_id"]
            for record in sample
        }

        remaining = [
            record
            for record in eligible
            if record["window_id"] not in selected_window_ids
        ]

        rng.shuffle(remaining)

        for record in remaining:
            if len(sample) >= SAMPLE_SIZE:
                break

            sample.append(record)

    sample.sort(
        key=lambda window: window["start_time"]
    )

    write_jsonl(
        sample,
        OUTPUT_PATH,
    )

    unique_sessions = len({
        window["session_id"]
        for window in sample
    })

    average_turns = (
        sum(
            window["window_turn_count"]
            for window in sample
        ) / len(sample)
        if sample
        else 0
    )

    print()
    print(f"Input windows:              {len(windows):,}")
    print(f"Eligible windows (>= {MIN_TURNS} turns): {len(eligible):,}")
    print(f"Sampled windows:            {len(sample):,}")
    print(f"Unique sessions represented:{unique_sessions:>6,}")
    print(f"Average turns/window:       {average_turns:.2f}")
    print(f"Chronological bins:         {len(bins)}")
    print(f"Random seed:                {RANDOM_SEED}")
    print(f"Output:                     {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
