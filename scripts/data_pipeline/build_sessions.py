import json
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

INPUT_PATH = Path("isaac_turns.jsonl")
OUTPUT_PATH = Path("isaac_sessions.jsonl")

# If this much time passes between consecutive turns,
# consider the next turn the start of a new conversation.
SESSION_GAP_HOURS = 3


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def parse_timestamp(value):
    """
    Convert an ISO-8601 timestamp string into a datetime object.
    """

    if value is None:
        return None

    return datetime.fromisoformat(value)


def hours_between(earlier, later):
    """
    Return the number of hours between two datetime objects.
    """

    if earlier is None or later is None:
        return 0

    return (later - earlier).total_seconds() / 3600


# ---------------------------------------------------------
# LOAD TURNS
# ---------------------------------------------------------

def load_turns(path):
    """
    Read isaac_turns.jsonl and return a list of turn objects.
    """

    turns = []

    with path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                turn = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            turns.append(turn)

    return turns


# ---------------------------------------------------------
# SESSION LOGIC
# ---------------------------------------------------------

def should_start_new_session(current_session, turn):
    """
    Decide whether a turn belongs to the current session
    or should begin a new session.
    """

    if current_session is None:
        return True

    previous_end = parse_timestamp(
        current_session["end_time"]
    )

    current_start = parse_timestamp(
        turn["start_time"]
    )

    gap_hours = hours_between(
        previous_end,
        current_start
    )

    return gap_hours > SESSION_GAP_HOURS


def make_session(session_id, turn):
    """
    Create a new session beginning with one turn.
    """

    return {
        "session_id": session_id,

        "start_time": turn["start_time"],
        "end_time": turn["end_time"],

        "turn_count": 1,

        "turn_ids": [
            turn["turn_id"]
        ],

        "speakers": [
            turn["speaker"]
        ],

        "turns": [
            turn
        ]
    }


def append_turn_to_session(session, turn):
    """
    Add a turn to an existing session.
    """

    session["end_time"] = turn["end_time"]

    session["turn_count"] += 1

    session["turn_ids"].append(
        turn["turn_id"]
    )

    if turn["speaker"] not in session["speakers"]:
        session["speakers"].append(
            turn["speaker"]
        )

    session["turns"].append(
        turn
    )


def build_sessions(turns):
    """
    Convert chronological turns into conversational sessions.
    """

    sessions = []
    current_session = None

    for turn in turns:

        if should_start_new_session(
            current_session,
            turn
        ):

            if current_session is not None:
                sessions.append(
                    current_session
                )

            current_session = make_session(
                session_id=len(sessions) + 1,
                turn=turn
            )

        else:

            append_turn_to_session(
                current_session,
                turn
            )

    if current_session is not None:
        sessions.append(
            current_session
        )

    return sessions


# ---------------------------------------------------------
# WRITE JSONL
# ---------------------------------------------------------

def write_jsonl(sessions, path):
    """
    Write one complete conversation session per JSONL line.
    """

    with path.open(
        "w",
        encoding="utf-8"
    ) as output:

        for session in sessions:

            json.dump(
                session,
                output,
                ensure_ascii=False
            )

            output.write("\n")


# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

def print_statistics(turns, sessions):

    total_turns = len(turns)
    total_sessions = len(sessions)

    turns_in_sessions = sum(
        session["turn_count"]
        for session in sessions
    )

    single_turn_sessions = sum(
        session["turn_count"] == 1
        for session in sessions
    )

    if total_sessions > 0:

        average_turns = (
            turns_in_sessions
            / total_sessions
        )

        largest_session = max(
            session["turn_count"]
            for session in sessions
        )

    else:

        average_turns = 0
        largest_session = 0

    print()
    print(
        f"Loaded turns:              "
        f"{total_turns:,}"
    )

    print(
        f"Turns placed in sessions:  "
        f"{turns_in_sessions:,}"
    )

    print(
        f"Total sessions:            "
        f"{total_sessions:,}"
    )

    print(
        f"Average turns per session: "
        f"{average_turns:.2f}"
    )

    print(
        f"Single-turn sessions:      "
        f"{single_turn_sessions:,}"
    )

    print(
        f"Largest session:           "
        f"{largest_session:,} turns"
    )

    print(
        f"Session gap threshold:     "
        f"{SESSION_GAP_HOURS} hours"
    )

    print(
        f"Output:                    "
        f"{OUTPUT_PATH}"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    turns = load_turns(
        INPUT_PATH
    )

    sessions = build_sessions(
        turns
    )

    write_jsonl(
        sessions,
        OUTPUT_PATH
    )

    print_statistics(
        turns,
        sessions
    )


if __name__ == "__main__":
    main()