import json
from datetime import datetime
from pathlib import Path

INPUT_PATH = Path("isaac_turns.jsonl")
SESSION_GAP_HOURS = 3
TOP_N = 15


def parse_timestamp(value):
    if value is None:
        return None
    return datetime.fromisoformat(value)


def hours_between(earlier, later):
    if earlier is None or later is None:
        return 0.0
    return (later - earlier).total_seconds() / 3600


def load_turns(path):
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


def start_session(session_id, turn):
    return {
        "session_id": session_id,
        "start_time": turn["start_time"],
        "end_time": turn["end_time"],
        "turn_ids": [turn["turn_id"]],
        "turn_count": 1,
        "max_internal_gap_hours": 0.0,
        "first_turn_text": turn.get("text", ""),
        "last_turn_text": turn.get("text", ""),
    }


def append_turn(session, turn, gap_hours):
    session["end_time"] = turn["end_time"]
    session["turn_ids"].append(turn["turn_id"])
    session["turn_count"] += 1
    if gap_hours > session["max_internal_gap_hours"]:
        session["max_internal_gap_hours"] = gap_hours
    session["last_turn_text"] = turn.get("text", "")


def build_sessions(turns):
    sessions = []
    if not turns:
        return sessions

    current_session = None
    previous_end = None

    for turn in turns:
        current_start = parse_timestamp(turn["start_time"])

        if current_session is None:
            current_session = start_session(1, turn)
        else:
            gap_hours = hours_between(previous_end, current_start)

            if gap_hours > SESSION_GAP_HOURS:
                sessions.append(current_session)
                current_session = start_session(len(sessions) + 1, turn)
            else:
                append_turn(current_session, turn, gap_hours)

        previous_end = parse_timestamp(turn["end_time"])

    if current_session is not None:
        sessions.append(current_session)

    return sessions


def add_duration_fields(sessions):
    for session in sessions:
        start = parse_timestamp(session["start_time"])
        end = parse_timestamp(session["end_time"])
        session["duration_hours"] = hours_between(start, end)


def shorten(text, length=80):
    if text is None:
        return ""
    text = text.replace("\n", " / ").strip()
    if len(text) <= length:
        return text
    return text[: length - 3] + "..."


def print_session(session):
    first_turn = session["turn_ids"][0]
    last_turn = session["turn_ids"][-1]

    print(f"Session {session['session_id']:,}")
    print(f"  Turns:              {session['turn_count']:,}")
    print(f"  Turn range:         {first_turn:,} - {last_turn:,}")
    print(f"  Duration:           {session['duration_hours']:.2f} hours")
    print(f"  Largest gap inside: {session['max_internal_gap_hours']:.2f} hours")
    print(f"  Start:              {session['start_time']}")
    print(f"  End:                {session['end_time']}")
    print(f"  First turn:         {shorten(session['first_turn_text'])}")
    print(f"  Last turn:          {shorten(session['last_turn_text'])}")
    print()


def main():
    turns = load_turns(INPUT_PATH)
    sessions = build_sessions(turns)
    add_duration_fields(sessions)

    print(f"Loaded {len(turns):,} turns")
    print(
        f"Built {len(sessions):,} sessions "
        f"using a {SESSION_GAP_HOURS}-hour inactivity threshold."
    )
    print()

    print(f"TOP {TOP_N} SESSIONS BY TURN COUNT")
    print("=" * 45)

    by_turn_count = sorted(
        sessions,
        key=lambda session: session["turn_count"],
        reverse=True,
    )

    for session in by_turn_count[:TOP_N]:
        print_session(session)

    print()
    print(f"TOP {TOP_N} SESSIONS BY TOTAL DURATION")
    print("=" * 45)

    by_duration = sorted(
        sessions,
        key=lambda session: session["duration_hours"],
        reverse=True,
    )

    for session in by_duration[:TOP_N]:
        print_session(session)


if __name__ == "__main__":
    main()
