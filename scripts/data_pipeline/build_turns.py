import json
from datetime import datetime
from pathlib import Path

INPUT_PATH = Path("isaac_messages_clean.jsonl")
OUTPUT_PATH = Path("isaac_turns.jsonl")

# If the same person sends more messages after a long pause, treat that as
# a new conversational turn even if the other person never replied.
MAX_SAME_SPEAKER_GAP_MINUTES = 10


def parse_timestamp(value):
    if value is None:
        return None
    return datetime.fromisoformat(value)


def minutes_between(earlier, later):
    if earlier is None or later is None:
        return 0

    return (later - earlier).total_seconds() / 60


def load_messages(path):
    messages = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            messages.append(message)

    return messages


def should_start_new_turn(current_turn, message):
    if current_turn is None:
        return True

    if message["sender"] != current_turn["speaker"]:
        return True

    previous_time = parse_timestamp(current_turn["end_time"])
    current_time = parse_timestamp(message["timestamp"])

    gap_minutes = minutes_between(previous_time, current_time)

    if gap_minutes > MAX_SAME_SPEAKER_GAP_MINUTES:
        return True

    return False


def make_turn(turn_id, message):
    text = message.get("text")

    return {
        "turn_id": turn_id,
        "speaker": message["sender"],
        "start_time": message["timestamp"],
        "end_time": message["timestamp"],
        "text": text if text is not None else "",
        "message_ids": [message["message_id"]],
        "message_count": 1,
        "attachments": list(message.get("attachments", [])),
    }


def append_message_to_turn(turn, message):
    text = message.get("text")

    if text:
        if turn["text"]:
            turn["text"] += "\n"
        turn["text"] += text

    turn["end_time"] = message["timestamp"]
    turn["message_ids"].append(message["message_id"])
    turn["message_count"] += 1
    turn["attachments"].extend(message.get("attachments", []))


def build_turns(messages):
    turns = []
    current_turn = None

    for message in messages:
        # Keep attachment-only messages, but skip records that contain neither
        # text nor attachment data.
        has_text = bool(message.get("text"))
        has_attachments = bool(message.get("attachments"))

        if not has_text and not has_attachments:
            continue

        if should_start_new_turn(current_turn, message):
            if current_turn is not None:
                turns.append(current_turn)

            current_turn = make_turn(
                turn_id=len(turns) + 1,
                message=message,
            )
        else:
            append_message_to_turn(current_turn, message)

    if current_turn is not None:
        turns.append(current_turn)

    return turns


def write_jsonl(turns, path):
    with path.open("w", encoding="utf-8") as output:
        for turn in turns:
            json.dump(turn, output, ensure_ascii=False)
            output.write("\n")


def main():
    messages = load_messages(INPUT_PATH)
    turns = build_turns(messages)
    write_jsonl(turns, OUTPUT_PATH)

    aaron_turns = sum(turn["speaker"] == "Aaron" for turn in turns)
    isaac_turns = sum(turn["speaker"] == "Isaac" for turn in turns)

    messages_in_turns = sum(turn["message_count"] for turn in turns)

    print(f"Loaded messages:          {len(messages):,}")
    print(f"Messages placed in turns: {messages_in_turns:,}")
    print(f"Total turns:              {len(turns):,}")
    print(f"Aaron turns:              {aaron_turns:,}")
    print(f"Isaac turns:              {isaac_turns:,}")
    print(f"Output:                   {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
