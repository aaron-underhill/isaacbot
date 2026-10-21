import json
import math
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_SESSIONS_PATH = PROJECT_ROOT / "data" / "splits" / "train_sessions.jsonl"
CLEAN_MESSAGES_PATH = PROJECT_ROOT / "data" / "processed" / "isaac_messages_clean.jsonl"

OUTPUT_JSON = PROJECT_ROOT / "data" / "analysis" / "train_surface_style_analysis.json"
OUTPUT_MD = (
    PROJECT_ROOT
    / "reports"
    / "communication_style"
    / "isaac_surface_communication_style_train_v0.1.md"
)

SELECTED_TOKENS = [
    "bro", "bruh", "lmao", "lol", "omg", "wtf", "fr", "girl", "dude",
    "pls", "plz", "nah", "yo", "ppl", "okok", "ima", "tysm", "lmk",
    "ty", "wat", "omfg", "goddamn", "yknow", "bout", "slay", "sumn",
    "ummm", "literally",
]

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]"
)

def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path} line {n}: {exc}") from exc
    return rows

def tokenize(text):
    return [x.lower().replace("’", "'") for x in WORD_RE.findall(text or "")]

def is_all_caps(text):
    letters = [c for c in (text or "") if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)

def detect_message_id_key(messages):
    if not messages:
        raise RuntimeError("No clean messages loaded.")
    candidates = ["message_id", "id", "rowid", "ROWID", "message_rowid"]
    first = messages[0]
    for key in candidates:
        if key in first:
            return key
    raise RuntimeError(
        "Could not detect message ID key. First record keys: "
        + ", ".join(sorted(first.keys()))
    )

def pct(n, d):
    return 100.0 * n / d if d else 0.0

def speaker_message_metrics(messages):
    out = {}
    for speaker in ("Aaron", "Isaac"):
        rows = [m for m in messages if m.get("speaker") == speaker]
        token_lists = [tokenize(m.get("text", "")) for m in rows]
        words = sum(len(tokens) for tokens in token_lists)
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        out[speaker] = {
            "messages": len(rows),
            "words": words,
            "avg_words_per_message": words / len(rows) if rows else 0.0,
            "all_caps_pct": pct(sum(is_all_caps(m.get("text", "")) for m in rows), len(rows)),
            "question_pct": pct(sum("?" in (m.get("text", "")) for m in rows), len(rows)),
            "exclamation_pct": pct(sum("!" in (m.get("text", "")) for m in rows), len(rows)),
            "emoji_pct": pct(sum(bool(EMOJI_RE.search(m.get("text", ""))) for m in rows), len(rows)),
            "url_pct": pct(sum(bool(URL_RE.search(m.get("text", ""))) for m in rows), len(rows)),
            "tokens": counter,
        }
    return out

def speaker_turn_metrics(turns):
    out = {}
    for speaker in ("Aaron", "Isaac"):
        rows = [t for t in turns if t.get("speaker") == speaker]
        words = sum(len(tokenize(t.get("text", ""))) for t in rows)
        message_count = sum(int(t.get("message_count", 0)) for t in rows)
        out[speaker] = {
            "turns": len(rows),
            "words": words,
            "avg_words_per_turn": words / len(rows) if rows else 0.0,
            "avg_messages_per_turn": message_count / len(rows) if rows else 0.0,
        }
    return out

def log2_rate_ratio(ic, iw, ac, aw, smoothing=0.5):
    ir = (ic + smoothing) / (iw + smoothing)
    ar = (ac + smoothing) / (aw + smoothing)
    return math.log2(ir / ar)

def distinctive_tokens(message_metrics, min_total_count=15, top_n=40):
    isaac = message_metrics["Isaac"]
    aaron = message_metrics["Aaron"]
    vocab = set(isaac["tokens"]) | set(aaron["tokens"])
    rows = []
    for token in vocab:
        ic = isaac["tokens"][token]
        ac = aaron["tokens"][token]
        if ic + ac < min_total_count:
            continue
        score = log2_rate_ratio(ic, isaac["words"], ac, aaron["words"])
        rows.append({
            "token": token,
            "score": score,
            "isaac_count": ic,
            "aaron_count": ac,
            "isaac_per_1k": 1000 * ic / isaac["words"] if isaac["words"] else 0.0,
            "aaron_per_1k": 1000 * ac / aaron["words"] if aaron["words"] else 0.0,
        })
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:top_n], list(reversed(rows[-top_n:]))

def main():
    train_sessions = load_jsonl(TRAIN_SESSIONS_PATH)
    clean_messages = load_jsonl(CLEAN_MESSAGES_PATH)

    train_turns = []
    train_message_ids = set()

    for session in train_sessions:
        turns = session.get("turns")
        if turns is None:
            raise RuntimeError("Expected embedded 'turns' in train_sessions.jsonl.")
        train_turns.extend(turns)
        for turn in turns:
            train_message_ids.update(turn.get("message_ids", []))

    message_id_key = detect_message_id_key(clean_messages)
    train_messages = [
        m for m in clean_messages
        if m.get(message_id_key) in train_message_ids
    ]

    message_metrics = speaker_message_metrics(train_messages)
    turn_metrics = speaker_turn_metrics(train_turns)
    isaac_distinctive, aaron_distinctive = distinctive_tokens(message_metrics)

    selected = {}
    for token in SELECTED_TOKENS:
        selected[token] = {}
        for speaker in ("Aaron", "Isaac"):
            metrics = message_metrics[speaker]
            count = metrics["tokens"][token]
            selected[token][speaker] = {
                "count": count,
                "per_1k": 1000 * count / metrics["words"] if metrics["words"] else 0.0,
            }

    result = {
        "scope": "TRAIN ONLY",
        "train_sessions": len(train_sessions),
        "train_turns": len(train_turns),
        "train_messages": len(train_messages),
        "message_metrics": {
            speaker: {k: v for k, v in metrics.items() if k != "tokens"}
            for speaker, metrics in message_metrics.items()
        },
        "turn_metrics": turn_metrics,
        "selected_tokens": selected,
        "isaac_distinctive": isaac_distinctive,
        "aaron_distinctive": aaron_distinctive,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    A = message_metrics["Aaron"]
    I = message_metrics["Isaac"]
    AT = turn_metrics["Aaron"]
    IT = turn_metrics["Isaac"]

    with OUTPUT_MD.open("w", encoding="utf-8") as f:
        f.write("# Isaac Surface Communication Style — TRAIN ONLY v0.1\n\n")
        f.write("Computed exclusively from training sessions. Validation/test sessions are excluded.\n\n")
        f.write(f"- Train sessions: **{len(train_sessions):,}**\n")
        f.write(f"- Train turns: **{len(train_turns):,}**\n")
        f.write(f"- Train messages: **{len(train_messages):,}**\n\n")
        f.write("| Metric | Aaron | Isaac |\n|---|---:|---:|\n")
        f.write(f"| Messages | {A['messages']:,} | {I['messages']:,} |\n")
        f.write(f"| Turns | {AT['turns']:,} | {IT['turns']:,} |\n")
        f.write(f"| Avg words/message | {A['avg_words_per_message']:.2f} | {I['avg_words_per_message']:.2f} |\n")
        f.write(f"| Avg words/turn | {AT['avg_words_per_turn']:.2f} | {IT['avg_words_per_turn']:.2f} |\n")
        f.write(f"| Avg messages/turn | {AT['avg_messages_per_turn']:.2f} | {IT['avg_messages_per_turn']:.2f} |\n")
        f.write(f"| ALL CAPS | {A['all_caps_pct']:.1f}% | {I['all_caps_pct']:.1f}% |\n")
        f.write(f"| Contains ? | {A['question_pct']:.1f}% | {I['question_pct']:.1f}% |\n")
        f.write(f"| Contains ! | {A['exclamation_pct']:.1f}% | {I['exclamation_pct']:.1f}% |\n")
        f.write(f"| Contains emoji | {A['emoji_pct']:.1f}% | {I['emoji_pct']:.1f}% |\n")
        f.write(f"| Contains URL | {A['url_pct']:.1f}% | {I['url_pct']:.1f}% |\n\n")
        f.write("## Selected tokens\n\n")
        f.write("| Token | Aaron count | Aaron /1k | Isaac count | Isaac /1k |\n|---|---:|---:|---:|---:|\n")
        for token in SELECTED_TOKENS:
            a = selected[token]["Aaron"]
            i = selected[token]["Isaac"]
            f.write(f"| `{token}` | {a['count']} | {a['per_1k']:.2f} | {i['count']} | {i['per_1k']:.2f} |\n")
        f.write("\n## Most Isaac-distinctive vocabulary\n\n")
        for row in isaac_distinctive[:25]:
            f.write(
                f"- `{row['token']}` — score {row['score']:.2f}; "
                f"Isaac {row['isaac_count']} ({row['isaac_per_1k']:.2f}/1k), "
                f"Aaron {row['aaron_count']} ({row['aaron_per_1k']:.2f}/1k)\n"
            )

    print()
    print("TRAIN-ONLY surface analysis complete.")
    print(f"Train sessions:  {len(train_sessions):,}")
    print(f"Train turns:     {len(train_turns):,}")
    print(f"Train messages:  {len(train_messages):,}")
    print(f"JSON:            {OUTPUT_JSON}")
    print(f"Markdown:        {OUTPUT_MD}")

if __name__ == "__main__":
    main()
