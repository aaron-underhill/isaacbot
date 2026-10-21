import json
from datetime import datetime
from pathlib import Path
from statistics import median


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

INPUT_PATH = Path("isaac_turns.jsonl")

# Session-gap thresholds we want to compare.
THRESHOLDS_HOURS = [
    0.5,
    1,
    2,
    3,
    4,
    6,
]


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def parse_timestamp(value):
    """
    Convert ISO-8601 timestamp string into a datetime object.
    """

    if value is None:
        return None

    return datetime.fromisoformat(value)


def hours_between(earlier, later):
    """
    Return elapsed hours between two datetime objects.
    """

    if earlier is None or later is None:
        return 0

    return (
        later - earlier
    ).total_seconds() / 3600


def percentile(values, percent):
    """
    Calculate a percentile using linear interpolation.

    Example:
        percentile(values, 90)
    returns the 90th percentile.
    """

    if not values:
        return 0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (
        percent / 100
    ) * (len(values) - 1)

    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(values) - 1
    )

    fraction = (
        position - lower_index
    )

    lower_value = values[lower_index]
    upper_value = values[upper_index]

    return (
        lower_value
        + fraction
        * (upper_value - lower_value)
    )


# ---------------------------------------------------------
# LOAD TURNS
# ---------------------------------------------------------

def load_turns(path):
    turns = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:
                turn = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

            turns.append(turn)

    return turns


# ---------------------------------------------------------
# BUILD SESSION SIZES
# ---------------------------------------------------------

def get_session_sizes(
    turns,
    threshold_hours
):
    """
    Group turns using a particular inactivity threshold.

    We only need session sizes here, not complete session
    objects, because this script is for comparison.
    """

    if not turns:
        return []

    session_sizes = []

    current_size = 1

    previous_end = parse_timestamp(
        turns[0]["end_time"]
    )

    for turn in turns[1:]:

        current_start = parse_timestamp(
            turn["start_time"]
        )

        gap = hours_between(
            previous_end,
            current_start
        )

        if gap > threshold_hours:

            session_sizes.append(
                current_size
            )

            current_size = 1

        else:

            current_size += 1

        previous_end = parse_timestamp(
            turn["end_time"]
        )

    session_sizes.append(
        current_size
    )

    return session_sizes


# ---------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------

def analyze_threshold(
    turns,
    threshold_hours
):
    sizes = get_session_sizes(
        turns,
        threshold_hours
    )

    total_sessions = len(sizes)

    total_turns = sum(sizes)

    average = (
        total_turns / total_sessions
        if total_sessions
        else 0
    )

    single_turn_sessions = sum(
        size == 1
        for size in sizes
    )

    single_turn_percent = (
        single_turn_sessions
        / total_sessions
        * 100
        if total_sessions
        else 0
    )

    return {
        "threshold": threshold_hours,
        "sessions": total_sessions,
        "average": average,
        "median": median(sizes),
        "single_turns": single_turn_sessions,
        "single_percent": single_turn_percent,
        "p90": percentile(sizes, 90),
        "p95": percentile(sizes, 95),
        "p99": percentile(sizes, 99),
        "largest": max(sizes) if sizes else 0,
    }


# ---------------------------------------------------------
# PRINT TABLE
# ---------------------------------------------------------

def print_results(results):

    print()

    header = (
        f"{'Gap':>7} "
        f"{'Sessions':>10} "
        f"{'Avg':>8} "
        f"{'Median':>8} "
        f"{'Singles':>10} "
        f"{'Single %':>9} "
        f"{'P90':>7} "
        f"{'P95':>7} "
        f"{'P99':>7} "
        f"{'Max':>7}"
    )

    print(header)

    print(
        "-" * len(header)
    )

    for result in results:

        print(
            f"{result['threshold']:>6.1f}h "
            f"{result['sessions']:>10,} "
            f"{result['average']:>8.2f} "
            f"{result['median']:>8.1f} "
            f"{result['single_turns']:>10,} "
            f"{result['single_percent']:>8.1f}% "
            f"{result['p90']:>7.1f} "
            f"{result['p95']:>7.1f} "
            f"{result['p99']:>7.1f} "
            f"{result['largest']:>7,}"
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    turns = load_turns(
        INPUT_PATH
    )

    print(
        f"Loaded {len(turns):,} turns "
        f"from {INPUT_PATH}"
    )

    results = []

    for threshold in THRESHOLDS_HOURS:

        result = analyze_threshold(
            turns,
            threshold
        )

        results.append(
            result
        )

    print_results(
        results
    )


if __name__ == "__main__":
    main()