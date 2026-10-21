import json
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_PATH = Path("isaac_sessions.jsonl")
OUTPUT_PATH = Path("isaac_behavioral_windows.jsonl")

# Nominal number of turns per analysis window.
TARGET_TURNS_PER_WINDOW = 16

# Number of turns intentionally repeated between neighboring windows.
OVERLAP_TURNS = 4


# =========================================================
# IO
# =========================================================

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} on line "
                    f"{line_number}: {exc}"
                ) from exc


def write_jsonl_record(file, record):
    json.dump(
        record,
        file,
        ensure_ascii=False,
    )

    file.write("\n")


# =========================================================
# WINDOW BOUNDARY HELPERS
# =========================================================

def adjust_start_for_context(turns, start_index):
    """
    If a proposed window begins with an Isaac turn that directly
    follows an Aaron turn, shift the window one turn backward.

    This keeps the immediate Aaron prompt/context attached to the
    Isaac response whenever possible.
    """

    if start_index <= 0:
        return start_index

    current = turns[start_index]
    previous = turns[start_index - 1]

    if (
        current.get("speaker") == "Isaac"
        and previous.get("speaker") == "Aaron"
    ):
        return start_index - 1

    return start_index


def extend_end_for_response(turns, end_index):
    """
    end_index is exclusive.

    If a proposed window would end immediately after an Aaron turn
    while the next turn is Isaac, extend the window so that Isaac's
    response is not cut away from Aaron's prompt.

    If Isaac has multiple consecutive turns after Aaron, keep the
    whole immediate Isaac response streak.
    """

    if end_index >= len(turns):
        return end_index

    last_included = turns[end_index - 1]
    next_turn = turns[end_index]

    if not (
        last_included.get("speaker") == "Aaron"
        and next_turn.get("speaker") == "Isaac"
    ):
        return end_index

    while (
        end_index < len(turns)
        and turns[end_index].get("speaker") == "Isaac"
    ):
        end_index += 1

    return end_index


# =========================================================
# WINDOW CREATION
# =========================================================

def collect_message_ids(window_turns):
    message_ids = []

    for turn in window_turns:
        message_ids.extend(
            turn.get("message_ids", [])
        )

    return message_ids


def make_window(
    window_id,
    session,
    window_index_in_session,
    turns,
    start_index,
    end_index,
):
    window_turns = turns[
        start_index:end_index
    ]

    return {
        "window_id": window_id,
        "session_id": session["session_id"],
        "window_index_in_session": window_index_in_session,

        "session_turn_count": len(turns),
        "window_turn_count": len(window_turns),

        "start_turn_id": window_turns[0]["turn_id"],
        "end_turn_id": window_turns[-1]["turn_id"],

        "start_time": window_turns[0]["start_time"],
        "end_time": window_turns[-1]["end_time"],

        "turn_ids": [
            turn["turn_id"]
            for turn in window_turns
        ],

        "message_ids": collect_message_ids(
            window_turns
        ),

        "turns": window_turns,
    }


def build_windows_for_session(
    session,
    next_window_id,
):
    turns = session.get("turns", [])

    if not turns:
        return [], next_window_id

    windows = []

    # Short sessions stay intact as one window.
    if len(turns) <= TARGET_TURNS_PER_WINDOW:
        window = make_window(
            window_id=next_window_id,
            session=session,
            window_index_in_session=1,
            turns=turns,
            start_index=0,
            end_index=len(turns),
        )

        windows.append(window)

        return windows, next_window_id + 1

    nominal_start = 0
    window_index_in_session = 1

    while nominal_start < len(turns):

        # Preserve the immediate Aaron -> Isaac context at a window start.
        actual_start = adjust_start_for_context(
            turns,
            nominal_start,
        )

        nominal_end = min(
            nominal_start + TARGET_TURNS_PER_WINDOW,
            len(turns),
        )

        # Preserve Aaron -> Isaac response pairs at a window end.
        actual_end = extend_end_for_response(
            turns,
            nominal_end,
        )

        window = make_window(
            window_id=next_window_id,
            session=session,
            window_index_in_session=window_index_in_session,
            turns=turns,
            start_index=actual_start,
            end_index=actual_end,
        )

        windows.append(window)

        next_window_id += 1
        window_index_in_session += 1

        if actual_end >= len(turns):
            break

        # Intentional overlap gives the next window recent context.
        next_start = actual_end - OVERLAP_TURNS

        # Always make forward progress.
        if next_start <= nominal_start:
            next_start = nominal_start + 1

        nominal_start = next_start

    return windows, next_window_id


# =========================================================
# VALIDATION
# =========================================================

def count_boundary_response_cuts(session, windows):
    """
    Count cases where a window ends on Aaron while the immediately
    following session turn is Isaac.

    With our response-preserving extension, this should be zero.
    """

    turns = session.get("turns", [])

    turn_id_to_index = {
        turn["turn_id"]: index
        for index, turn in enumerate(turns)
    }

    cuts = 0

    for window in windows:
        last_turn_id = window["end_turn_id"]
        last_index = turn_id_to_index[last_turn_id]

        if last_index >= len(turns) - 1:
            continue

        last_turn = turns[last_index]
        next_turn = turns[last_index + 1]

        if (
            last_turn.get("speaker") == "Aaron"
            and next_turn.get("speaker") == "Isaac"
        ):
            cuts += 1

    return cuts


# =========================================================
# MAIN
# =========================================================

def main():
    sessions = list(
        load_jsonl(INPUT_PATH)
    )

    total_windows = 0
    total_window_turns = 0
    min_window_turns = None
    max_window_turns = 0

    short_sessions_kept_whole = 0
    multi_window_sessions = 0
    response_pair_cuts = 0

    next_window_id = 1

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for session in sessions:
            session_turns = session.get(
                "turns",
                []
            )

            windows, next_window_id = (
                build_windows_for_session(
                    session,
                    next_window_id,
                )
            )

            if len(session_turns) <= TARGET_TURNS_PER_WINDOW:
                short_sessions_kept_whole += 1
            elif len(windows) > 1:
                multi_window_sessions += 1

            response_pair_cuts += (
                count_boundary_response_cuts(
                    session,
                    windows,
                )
            )

            for window in windows:
                write_jsonl_record(
                    output_file,
                    window,
                )

                size = window[
                    "window_turn_count"
                ]

                total_windows += 1
                total_window_turns += size

                if (
                    min_window_turns is None
                    or size < min_window_turns
                ):
                    min_window_turns = size

                if size > max_window_turns:
                    max_window_turns = size

    average_window_turns = (
        total_window_turns / total_windows
        if total_windows
        else 0.0
    )

    print()
    print(
        f"Sessions loaded:             "
        f"{len(sessions):,}"
    )

    print(
        f"Behavioral windows created:  "
        f"{total_windows:,}"
    )

    print(
        f"Target turns/window:         "
        f"{TARGET_TURNS_PER_WINDOW}"
    )

    print(
        f"Overlap turns:               "
        f"{OVERLAP_TURNS}"
    )

    print(
        f"Average actual window size:  "
        f"{average_window_turns:.2f} turns"
    )

    print(
        f"Smallest window:             "
        f"{(min_window_turns or 0):,} turns"
    )

    print(
        f"Largest window:              "
        f"{max_window_turns:,} turns"
    )

    print(
        f"Short sessions kept whole:   "
        f"{short_sessions_kept_whole:,}"
    )

    print(
        f"Sessions split into windows: "
        f"{multi_window_sessions:,}"
    )

    print(
        f"Aaron->Isaac boundary cuts:  "
        f"{response_pair_cuts:,}"
    )

    print(
        f"Output:                      "
        f"{OUTPUT_PATH}"
    )

    print()
    print(
        "Expected Aaron->Isaac boundary cuts: 0"
    )


if __name__ == "__main__":
    main()
