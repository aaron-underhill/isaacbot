import os
import sqlite3
import json
import re

from pathlib import Path
from datetime import datetime, timedelta, timezone

from typedstream import unarchive_from_data
from typedstream.archiving import TypedValue
from typedstream.types.foundation import NSString, NSMutableString


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"

# Keep the original export untouched while we test attributedBody decoding.
OUTPUT_PATH = Path("isaac_messages_raw_v2.jsonl")

MY_NAME = "Aaron"
BROTHER_NAME = "Isaac"

# Isaac's phone number / iMessage email address(es) are private, so they
# live in the project's .env file rather than in code:
#
#     ISAACBOT_TARGET_HANDLES=555-123-4567,someone@example.com
#
# Separate multiple Messages identities with commas. If the variable is
# missing, the script lists every direct-message participant it finds.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

TARGET_HANDLES = [
    handle.strip()
    for handle in os.getenv("ISAACBOT_TARGET_HANDLES", "").split(",")
    if handle.strip()
]


# Apple's Core Data epoch begins on January 1, 2001.
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def normalize_handle(handle):
    """
    Normalize phone numbers and email addresses so that differently
    formatted versions of the same phone number still match.
    """

    if not handle:
        return None

    handle = handle.strip().lower()

    if "@" in handle:
        return handle

    digits = re.sub(r"\D", "", handle)

    # Compare U.S. phone numbers primarily by their last 10 digits.
    if len(digits) >= 10:
        return digits[-10:]

    return digits


def apple_timestamp_to_iso(value):
    """
    Convert Apple's Core Data timestamp into an ISO 8601 timestamp.

    Modern Messages databases generally store nanoseconds since
    2001-01-01, while some older databases may store seconds.
    """

    if value is None:
        return None

    value = int(value)

    if abs(value) > 1_000_000_000_000:
        seconds = value / 1_000_000_000
    else:
        seconds = value

    timestamp = APPLE_EPOCH + timedelta(seconds=seconds)

    # Convert UTC to the Mac's local timezone.
    return timestamp.astimezone().isoformat()


def clean_decoded_text(text):
    """
    Remove a couple of formatting artifacts that can appear in decoded
    NSAttributedString message bodies.
    """

    if text is None:
        return None

    # U+FFFC is the Unicode object-replacement character commonly used
    # as an attachment marker inside attributed strings. Attachment
    # metadata is exported separately, so it is not useful in the text.
    text = text.replace("\ufffc", "")

    # Some decoded strings can begin with replacement characters.
    text = text.lstrip("\ufffd")

    return text


def decode_attributed_body(blob):
    """
    Recover plain text from Apple's message.attributedBody field.

    attributedBody is commonly an NSArchiver typedstream containing an
    NSAttributedString / NSMutableAttributedString. pytypedstream decodes
    that archive so we can retrieve the NSString containing the message.

    Returns:
        str | None: decoded message text, or None if decoding fails.
    """

    if blob is None:
        return None

    try:
        attributed = unarchive_from_data(blob)
    except Exception:
        return None

    # Occasionally a decoded value may itself be an NSString.
    if isinstance(attributed, (NSString, NSMutableString)):
        return clean_decoded_text(attributed.value)

    # Normal Messages attributedBody records should be an
    # NSAttributedString or NSMutableAttributedString archive.
    clazz = getattr(attributed, "clazz", None)
    class_name = getattr(clazz, "name", None)

    if class_name not in (
        b"NSAttributedString",
        b"NSMutableAttributedString",
    ):
        return None

    contents = getattr(attributed, "contents", None)

    if contents is None:
        return None

    # The underlying string is stored as an NSString/NSMutableString
    # within the attributed string's archived contents.
    for item in contents:
        value = item.value if isinstance(item, TypedValue) else item

        if isinstance(value, (NSString, NSMutableString)):
            return clean_decoded_text(value.value)

    return None


# ---------------------------------------------------------
# OPEN DATABASE READ-ONLY
# ---------------------------------------------------------

connection_string = f"file:{DB_PATH}?mode=ro"

conn = sqlite3.connect(
    connection_string,
    uri=True,
)

conn.row_factory = sqlite3.Row


# ---------------------------------------------------------
# FIND ONE-ON-ONE CHATS
# ---------------------------------------------------------

direct_chats = conn.execute(
    """
    SELECT
        c.ROWID AS chat_id,
        c.chat_identifier,
        c.display_name,
        h.id AS participant
    FROM chat AS c

    JOIN chat_handle_join AS chj
        ON chj.chat_id = c.ROWID

    JOIN handle AS h
        ON h.ROWID = chj.handle_id

    WHERE c.ROWID IN (
        SELECT chat_id
        FROM chat_handle_join
        GROUP BY chat_id
        HAVING COUNT(*) = 1
    )
    """
).fetchall()


# ---------------------------------------------------------
# IF NO TARGET WAS PROVIDED, SHOW POSSIBLE HANDLES
# ---------------------------------------------------------

if not TARGET_HANDLES:
    print("\nDirect-message participants found:\n")

    participants = {}

    for row in direct_chats:
        participant = row["participant"]
        participants[participant] = participants.get(participant, 0) + 1

    for participant in sorted(participants):
        print(participant)

    print(
        "\nAdd Isaac's phone number/email to ISAACBOT_TARGET_HANDLES "
        "in your .env file and run this script again."
    )

    conn.close()
    raise SystemExit


# ---------------------------------------------------------
# FIND ISAAC'S CHAT IDS
# ---------------------------------------------------------

target_handles = {
    normalize_handle(handle)
    for handle in TARGET_HANDLES
}

matching_chats = []

for row in direct_chats:
    participant = normalize_handle(row["participant"])

    if participant in target_handles:
        matching_chats.append(row)


if not matching_chats:
    conn.close()
    raise RuntimeError("No matching direct conversation was found.")


chat_ids = {
    row["chat_id"]
    for row in matching_chats
}


print("\nMatched chats:")

for row in matching_chats:
    print(
        row["chat_id"],
        row["participant"],
        row["chat_identifier"],
    )


# ---------------------------------------------------------
# FETCH MESSAGES
# ---------------------------------------------------------

placeholders = ",".join("?" for _ in chat_ids)

query = f"""
SELECT
    m.ROWID AS message_id,
    m.guid,
    m.text,
    m.attributedBody AS attributed_body,
    m.date,
    m.is_from_me,
    m.service,
    h.id AS handle,
    cmj.chat_id

FROM message AS m

JOIN chat_message_join AS cmj
    ON cmj.message_id = m.ROWID

LEFT JOIN handle AS h
    ON h.ROWID = m.handle_id

WHERE cmj.chat_id IN ({placeholders})

ORDER BY
    m.date ASC,
    m.ROWID ASC
"""

rows = conn.execute(
    query,
    tuple(chat_ids),
).fetchall()


# ---------------------------------------------------------
# FETCH ATTACHMENT METADATA
# ---------------------------------------------------------

attachment_query = f"""
SELECT
    maj.message_id,
    a.ROWID AS attachment_id,
    a.filename,
    a.mime_type,
    a.transfer_name

FROM message_attachment_join AS maj

JOIN attachment AS a
    ON a.ROWID = maj.attachment_id

JOIN chat_message_join AS cmj
    ON cmj.message_id = maj.message_id

WHERE cmj.chat_id IN ({placeholders})
"""

attachment_rows = conn.execute(
    attachment_query,
    tuple(chat_ids),
).fetchall()

attachments_by_message = {}

for attachment in attachment_rows:
    message_id = attachment["message_id"]

    attachments_by_message.setdefault(
        message_id,
        [],
    ).append(
        {
            "attachment_id": attachment["attachment_id"],
            "filename": attachment["filename"],
            "mime_type": attachment["mime_type"],
            "transfer_name": attachment["transfer_name"],
        }
    )


# ---------------------------------------------------------
# WRITE JSONL
# ---------------------------------------------------------

seen_messages = set()

exported = 0
plain_text_count = 0
attributed_text_count = 0
decode_failed_count = 0
no_text_data_count = 0


with OUTPUT_PATH.open("w", encoding="utf-8") as output:
    for row in rows:
        message_id = row["message_id"]

        # Avoid possible duplicates if Messages associates one message
        # with multiple chat records.
        if message_id in seen_messages:
            continue

        seen_messages.add(message_id)

        is_from_me = bool(row["is_from_me"])
        sender = MY_NAME if is_from_me else BROTHER_NAME

        text = row["text"]
        text_source = None

        if text is not None:
            text_source = "text"
            plain_text_count += 1

        elif row["attributed_body"] is not None:
            text = decode_attributed_body(row["attributed_body"])

            if text is not None:
                text_source = "attributedBody"
                attributed_text_count += 1
            else:
                decode_failed_count += 1

        else:
            no_text_data_count += 1

        record = {
            "message_id": message_id,
            "guid": row["guid"],
            "timestamp": apple_timestamp_to_iso(row["date"]),
            "sender": sender,
            "is_from_me": is_from_me,
            "text": text,
            "text_source": text_source,
            "service": row["service"],
            "attachments": attachments_by_message.get(message_id, []),
        }

        json.dump(
            record,
            output,
            ensure_ascii=False,
        )
        output.write("\n")

        exported += 1


conn.close()


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

missing_text_count = decode_failed_count + no_text_data_count

print(f"\nExported {exported:,} messages to {OUTPUT_PATH}")
print(f"Plain message.text:              {plain_text_count:,}")
print(f"Recovered from attributedBody:  {attributed_text_count:,}")
print(f"attributedBody decode failures: {decode_failed_count:,}")
print(f"No text or attributedBody:      {no_text_data_count:,}")
print(f"Still missing text:             {missing_text_count:,}")
