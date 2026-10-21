import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_SESSIONS_PATH = PROJECT_ROOT / "data" / "splits" / "train_sessions.jsonl"
VALIDATION_SESSIONS_PATH = PROJECT_ROOT / "data" / "splits" / "validation_sessions.jsonl"
TEST_SESSIONS_PATH = PROJECT_ROOT / "data" / "splits" / "test_sessions.jsonl"

OBSERVATIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "behavioral_observations_pilot_v0.1.jsonl"
)
RAG_METADATA_PATH = PROJECT_ROOT / "data" / "rag" / "isaac_rag_metadata.jsonl"

def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def ids(path):
    return {row["session_id"] for row in load_jsonl(path)}

def main():
    train_ids = ids(TRAIN_SESSIONS_PATH)
    validation_ids = ids(VALIDATION_SESSIONS_PATH)
    test_ids = ids(TEST_SESSIONS_PATH)

    observations = load_jsonl(OBSERVATIONS_PATH)
    rag_metadata = load_jsonl(RAG_METADATA_PATH)

    observation_session_ids = {row["session_id"] for row in observations}
    rag_session_ids = {row["session_id"] for row in rag_metadata}

    bad_observations = observation_session_ids - train_ids
    bad_rag = rag_session_ids - train_ids

    print()
    print("Split integrity")
    print("---------------")
    print(f"Train sessions:       {len(train_ids):,}")
    print(f"Validation sessions:  {len(validation_ids):,}")
    print(f"Test sessions:        {len(test_ids):,}")
    print()
    print("Behavioral persona pilot")
    print("------------------------")
    print(f"Observation sessions: {len(observation_session_ids):,}")
    print(f"Outside train:        {len(bad_observations):,}")
    print()
    print("RAG index")
    print("---------")
    print(f"RAG sessions:         {len(rag_session_ids):,}")
    print(f"Outside train:        {len(bad_rag):,}")
    print()

    if bad_observations:
        print("ERROR: behavioral observations include non-training sessions:")
        print(sorted(bad_observations)[:20])

    if bad_rag:
        print("ERROR: RAG metadata includes non-training sessions:")
        print(sorted(bad_rag)[:20])

    if not bad_observations and not bad_rag:
        print("PASS: behavioral persona evidence and RAG are training-only.")

if __name__ == "__main__":
    main()
