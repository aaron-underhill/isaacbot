import json
import re
from collections import Counter
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_PATH = Path("isaac_messages_raw_v2.jsonl")

CLEAN_OUTPUT_PATH = Path("isaac_messages_clean.jsonl")
REACTIONS_OUTPUT_PATH = Path("isaac_reactions.jsonl")


# =========================================================
# HIGH-CONFIDENCE iMESSAGE TAPBACK / REACTION PATTERNS
# =========================================================

# Classic Messages-generated tapback text, for example:
#
#   Loved “that was hilarious”
#   Liked an image
#   Laughed at “bro”
#   Emphasized an attachment
#   Questioned “what”
#
# The regex is deliberately conservative: it must match the
# entire message and the target must look like a quoted message
# or a known attachment description.

TARGET_PATTERN = (
    r'(?:'
    r'an image|'
    r'an attachment|'
    r'a photo|'
    r'a video|'
    r'an audio message|'
    r'a message|'
    r'“.*”|'
    r'".*"'
    r')'
)

REACTION_RE = re.compile(
    rf'^(Loved|Liked|Disliked|Laughed at|Emphasized|Questioned)\s+'
    rf'({TARGET_PATTERN})\s*$',
    re.IGNORECASE | re.DOTALL,
)

REMOVED_REACTION_RE = re.compile(
    rf'^Removed\s+'
    rf'(a heart|a like|a dislike|a laugh|an emphasis|a question mark)\s+'
    rf'from\s+({TARGET_PATTERN})\s*$',
    re.IGNORECASE | re.DOTALL,
)

# Some newer Messages versions can render reactions like:
#
#   Reacted ❤️ to “message”
#
# Again, this must match the whole message and point at a
# recognizable quoted/attachment target.
REACTED_RE = re.compile(
    rf'^Reacted\s+(.+?)\s+to\s+({TARGET_PATTERN})\s*$',
    re.IGNORECASE | re.DOTALL,
)


REACTION_NAME_MAP = {
    "loved": "love",
    "liked": "like",
    "disliked": "dislike",
    "laughed at": "laugh",
    "emphasized": "emphasis",
    "questioned": "question",
}

REMOVED_NAME_MAP = {
    "a heart": "love",
    "a like": "like",
    "a dislike": "dislike",
    "a laugh": "laugh",
    "an emphasis": "emphasis",
    "a question mark": "question",
}


# =========================================================
# HELPERS
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


def clean_visible_text(text):
    """
    Remove known invisible/object placeholder characters without
    correcting spelling, slang, capitalization, punctuation, etc.
    """

    if text is None:
        return None

    # U+FFFC = OBJECT REPLACEMENT CHARACTER.
    # Messages often uses it as a placeholder for rich content.
    text = text.replace("\ufffc", "")

    # U+FFFD = Unicode replacement character. If it appears at an
    # edge because of decoding noise, trim it there. We do not
    # otherwise rewrite the user's text.
    text = text.strip("\ufffd")

    return text


def classify_reaction(text):
    """
    Return reaction metadata for a high-confidence Messages-generated
    reaction artifact, otherwise return None.
    """

    if not text:
        return None

    match = REACTION_RE.fullmatch(text)

    if match:
        raw_name = match.group(1).lower()
        target = match.group(2)

        return {
            "artifact_type": "imessage_reaction",
            "action": "add",
            "reaction": REACTION_NAME_MAP[raw_name],
            "target": target,
        }

    match = REMOVED_REACTION_RE.fullmatch(text)

    if match:
        raw_name = match.group(1).lower()
        target = match.group(2)

        return {
            "artifact_type": "imessage_reaction",
            "action": "remove",
            "reaction": REMOVED_NAME_MAP[raw_name],
            "target": target,
        }

    match = REACTED_RE.fullmatch(text)

    if match:
        reaction_symbol = match.group(1)
        target = match.group(2)

        return {
            "artifact_type": "imessage_reaction",
            "action": "add",
            "reaction": reaction_symbol,
            "target": target,
        }

    return None


def write_jsonl_record(file, record):
    json.dump(
        record,
        file,
        ensure_ascii=False,
    )

    file.write("\n")


# =========================================================
# MAIN
# =========================================================

def main():
    input_count = 0
    clean_count = 0
    reaction_count = 0
    placeholder_cleaned_count = 0

    reactions_by_speaker = Counter()
    reactions_by_type = Counter()

    reaction_examples = []

    with (
        CLEAN_OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
        ) as clean_file,
        REACTIONS_OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
        ) as reaction_file,
    ):

        for message in load_jsonl(INPUT_PATH):
            input_count += 1

            original_text = message.get("text")
            cleaned_text = clean_visible_text(
                original_text
            )

            if cleaned_text != original_text:
                placeholder_cleaned_count += 1

            message["text"] = cleaned_text

            reaction = classify_reaction(
                cleaned_text
            )

            if reaction is not None:
                reaction_count += 1

                speaker = message.get(
                    "sender",
                    "Unknown",
                )

                reactions_by_speaker[speaker] += 1

                reaction_key = (
                    f"{reaction['action']}:"
                    f"{reaction['reaction']}"
                )

                reactions_by_type[
                    reaction_key
                ] += 1

                saved_reaction = {
                    **message,
                    **reaction,
                }

                write_jsonl_record(
                    reaction_file,
                    saved_reaction,
                )

                if len(reaction_examples) < 12:
                    reaction_examples.append(
                        {
                            "message_id": message.get(
                                "message_id"
                            ),
                            "sender": speaker,
                            "text": cleaned_text,
                        }
                    )

                # Do NOT put Messages-generated reaction text into
                # the dialogue corpus.
                continue

            write_jsonl_record(
                clean_file,
                message,
            )

            clean_count += 1

    print()
    print(
        f"Input messages:             "
        f"{input_count:,}"
    )

    print(
        f"Dialogue messages kept:     "
        f"{clean_count:,}"
    )

    print(
        f"Reaction artifacts removed: "
        f"{reaction_count:,}"
    )

    print(
        f"Texts with placeholder chars cleaned: "
        f"{placeholder_cleaned_count:,}"
    )

    print()
    print("Reactions by speaker:")

    for speaker, count in reactions_by_speaker.most_common():
        print(
            f"  {speaker:<10} {count:>6,}"
        )

    print()
    print("Reactions by type:")

    for reaction_type, count in reactions_by_type.most_common():
        print(
            f"  {reaction_type:<24} "
            f"{count:>6,}"
        )

    if reaction_examples:
        print()
        print(
            "Sample removed reaction artifacts "
            "(inspect these):"
        )

        for example in reaction_examples:
            text = (
                example["text"]
                .replace("\n", " / ")
            )

            if len(text) > 100:
                text = text[:97] + "..."

            print(
                f"  message_id={example['message_id']} "
                f"{example['sender']}: {text}"
            )

    print()
    print(
        f"Clean dialogue: {CLEAN_OUTPUT_PATH}"
    )

    print(
        f"Saved reactions: {REACTIONS_OUTPUT_PATH}"
    )

    print()
    print(
        "Original raw export was left untouched."
    )


if __name__ == "__main__":
    main()
