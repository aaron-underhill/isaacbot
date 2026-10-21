import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

OBSERVATIONS_PATH = Path("processed/behavioral_observations_pilot_v0.1.jsonl")
WINDOWS_PATH = Path("processed/behavioral_sample_v0.1.jsonl")

JSON_OUTPUT_PATH = Path("behavioral_trait_candidates_v0.1.json")
MARKDOWN_OUTPUT_PATH = Path("behavioral_trait_candidates_v0.1.md")


# =========================================================
# TRAIT MAPPING
# =========================================================
#
# This is intentionally explicit rather than "magical."
#
# Each low-level observed behavior is assigned to a broader candidate
# behavioral trait. As our observation vocabulary grows, we can edit
# these mappings and regenerate the report without changing the source
# observations.
#
# Any behavior not listed here is preserved under "Unmapped behaviors"
# rather than silently discarded.
# =========================================================

TRAITS = {
    "supportive_responsiveness": {
        "title": "Supportive Responsiveness",
        "summary": (
            "Isaac often reduces pressure, makes room for Aaron's competing "
            "obligations, or shifts into practical support when Aaron is "
            "clearly stressed, overloaded, or unavailable."
        ),
        "behaviors": {
            "backs_off_when_aaron_is_overloaded",
            "prioritizes_aarons_immediate_responsibility",
            "offers_low_pressure_availability",
            "respects_busyness_after_initial_request",
        },
    },

    "direct_practical_communication": {
        "title": "Direct Practical Communication",
        "summary": (
            "Isaac tends to communicate practical needs and urgent logistics "
            "directly, becoming especially terse or instructional when time "
            "pressure is high."
        ),
        "behaviors": {
            "becomes_highly_direct_and_instructional",
            "states_practical_needs_directly",
            "urgent_requests_can_be_direct_but_grateful",
            "signals_immediate_intent_to_respond",
        },
    },

    "exploratory_advice_and_reasoning": {
        "title": "Exploratory Advice and Reasoning",
        "summary": (
            "Isaac can give direct recommendations while still asking for more "
            "context, considering counterarguments, qualifying uncertainty, "
            "and revising his view when new information appears."
        ),
        "behaviors": {
            "offers_direct_but_nonabsolute_advice",
            "asks_clarifying_questions_before_concluding",
            "considers_counterarguments_and_possible_upsides",
            "occasionally_marks_uncertainty_about_memory",
            "updates_and_signals_uncertainty",
            "distinguishes_intent_from_impact",
        },
    },

    "protective_validation": {
        "title": "Protective Validation",
        "summary": (
            "When Isaac interprets another person's behavior as controlling, "
            "unfair, rude, or harmful to Aaron, he can become strongly "
            "validating and more categorical in his judgment."
        ),
        "behaviors": {
            "becomes_strongly_validating_when_he_perceives_control",
            "connects_current_conflict_to_prior_patterns",
            "strongly_validates_shared_negative_family_experiences",
        },
    },

    "collaborative_humor_and_teasing": {
        "title": "Collaborative Humor and Teasing",
        "summary": (
            "Isaac frequently participates in shared absurdist humor, quick "
            "roasting, exaggerated mock resentment, and joking tonal pivots, "
            "including during otherwise serious conversations."
        ),
        "behaviors": {
            "collaborative_absurdist_banter",
            "abrupt_absurdity_after_tension",
            "tolerates_playful_teasing_but_pushes_back",
            "uses_brief_teasing_inside_serious_conversations",
            "uses_mock_resentment_when_aaron_is_slow_to_help",
            "adds_roasting_even_while_making_a_serious_point",
            "uses_dark_or_ironic_humor_in_painful_topics",
        },
    },

    "aaron_as_social_sounding_board": {
        "title": "Aaron as a Social Sounding Board",
        "summary": (
            "Isaac often uses Aaron as an immediate source of judgment, advice, "
            "or social mediation when Isaac is uncertain about another person "
            "or an interaction."
        ),
        "behaviors": {
            "uses_aaron_as_social_intermediary",
            "seeks_aaron_as_real_time_social_sounding_board",
            "invokes_reciprocity_when_requesting_help",
        },
    },

    "shared_family_processing": {
        "title": "Shared Family Processing",
        "summary": (
            "Isaac and Aaron sometimes co-construct interpretations of shared "
            "family experiences. Isaac contributes his own memories, emotional "
            "reactions, and concise judgments to the shared narrative."
        ),
        "behaviors": {
            "strongly_validates_shared_negative_family_experiences",
            "openly_expresses_bitterness_and_nonforgiveness",
            "co_constructs_shared_interpretation_of_past_events",
            "uses_dark_or_ironic_humor_in_painful_topics",
        },
    },

    "enthusiastic_mutual_hype": {
        "title": "Enthusiastic Mutual Hype",
        "summary": (
            "Isaac can strongly mirror Aaron's excitement and reinforce future-"
            "oriented enthusiasm, especially around goals, careers, or shared "
            "interests."
        ),
        "behaviors": {
            "mirrors_hype_and_future_oriented_encouragement",
        },
    },

    "detailed_frustration_explanation": {
        "title": "Detailed Frustration Explanation",
        "summary": (
            "When explaining a personal frustration, Isaac may give concrete "
            "procedural details and explain why he sees the situation as unfair "
            "or mishandled."
        ),
        "behaviors": {
            "explains_frustration_with_concrete_detail",
        },
    },

    "low_stakes_nonadversarial_disagreement": {
        "title": "Low-Stakes Nonadversarial Disagreement",
        "summary": (
            "In low-stakes opinion discussions, Isaac can disagree or qualify "
            "Aaron's position without turning the exchange adversarial."
        ),
        "behaviors": {
            "lightweight_qualified_disagreement",
        },
    },
}


CONFIDENCE_WEIGHTS = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


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
                    f"Invalid JSON in {path} on line {line_number}: {exc}"
                ) from exc

    return records


# =========================================================
# HELPERS
# =========================================================

def year_from_iso(timestamp):
    if not timestamp:
        return None

    try:
        return datetime.fromisoformat(timestamp).year
    except ValueError:
        return None


def strength_label(observation_count, session_count, year_count):
    """
    A deliberately conservative V0.1 heuristic.

    This is NOT a psychological confidence score.
    It only describes how broadly the pattern is supported
    within this pilot sample.
    """

    if (
        observation_count >= 4
        and session_count >= 3
        and year_count >= 2
    ):
        return "strong"

    if (
        observation_count >= 2
        and session_count >= 2
    ):
        return "moderate"

    return "preliminary"


def mean_confidence_weight(observations):
    if not observations:
        return 0.0

    weights = [
        CONFIDENCE_WEIGHTS.get(
            observation.get("confidence", "low"),
            1,
        )
        for observation in observations
    ]

    return sum(weights) / len(weights)


def unique_preserving_order(values):
    seen = set()
    result = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


# =========================================================
# AGGREGATION
# =========================================================

def build_window_metadata(windows):
    metadata = {}

    for window in windows:
        metadata[window["window_id"]] = {
            "session_id": window.get("session_id"),
            "start_time": window.get("start_time"),
            "end_time": window.get("end_time"),
            "year": year_from_iso(
                window.get("start_time")
            ),
        }

    return metadata


def build_behavior_to_traits():
    behavior_to_traits = defaultdict(list)

    for trait_id, trait in TRAITS.items():
        for behavior in trait["behaviors"]:
            behavior_to_traits[behavior].append(
                trait_id
            )

    return behavior_to_traits


def aggregate(observations, window_metadata):
    behavior_to_traits = build_behavior_to_traits()

    grouped = {
        trait_id: []
        for trait_id in TRAITS
    }

    unmapped = []

    for observation in observations:
        behavior = observation.get("behavior")

        trait_ids = behavior_to_traits.get(
            behavior,
            []
        )

        if not trait_ids:
            unmapped.append(observation)
            continue

        for trait_id in trait_ids:
            grouped[trait_id].append(
                observation
            )

    trait_results = []

    for trait_id, trait_definition in TRAITS.items():
        trait_observations = grouped[
            trait_id
        ]

        if not trait_observations:
            continue

        session_ids = sorted({
            observation.get("session_id")
            for observation in trait_observations
            if observation.get("session_id") is not None
        })

        window_ids = sorted({
            observation.get("window_id")
            for observation in trait_observations
            if observation.get("window_id") is not None
        })

        years = set()

        for observation in trait_observations:
            window_id = observation.get(
                "window_id"
            )

            metadata = window_metadata.get(
                window_id,
                {}
            )

            year = metadata.get("year")

            if year is not None:
                years.add(year)

        years = sorted(years)

        category_counts = Counter(
            observation.get(
                "category",
                "unknown",
            )
            for observation in trait_observations
        )

        confidence_counts = Counter(
            observation.get(
                "confidence",
                "unknown",
            )
            for observation in trait_observations
        )

        scopes = unique_preserving_order([
            observation.get("scope")
            for observation in trait_observations
            if observation.get("scope")
        ])

        evidence_turn_ids = sorted({
            turn_id
            for observation in trait_observations
            for turn_id in observation.get(
                "evidence_turn_ids",
                []
            )
        })

        strength = strength_label(
            observation_count=len(
                trait_observations
            ),
            session_count=len(
                session_ids
            ),
            year_count=len(
                years
            ),
        )

        result = {
            "trait_id": trait_id,
            "title": trait_definition["title"],
            "summary": trait_definition["summary"],

            "support_strength": strength,

            "support": {
                "observation_count": len(
                    trait_observations
                ),
                "unique_sessions": len(
                    session_ids
                ),
                "unique_windows": len(
                    window_ids
                ),
                "years_observed": years,
                "year_count": len(years),

                "mean_confidence_weight": round(
                    mean_confidence_weight(
                        trait_observations
                    ),
                    2,
                ),

                "confidence_counts": dict(
                    confidence_counts
                ),

                "category_counts": dict(
                    category_counts
                ),
            },

            "contexts": scopes,

            "session_ids": session_ids,
            "window_ids": window_ids,
            "evidence_turn_ids": evidence_turn_ids,

            "observations": [
                {
                    "observation_id": observation.get(
                        "observation_id"
                    ),
                    "window_id": observation.get(
                        "window_id"
                    ),
                    "session_id": observation.get(
                        "session_id"
                    ),
                    "category": observation.get(
                        "category"
                    ),
                    "behavior": observation.get(
                        "behavior"
                    ),
                    "description": observation.get(
                        "description"
                    ),
                    "confidence": observation.get(
                        "confidence"
                    ),
                    "scope": observation.get(
                        "scope"
                    ),
                    "evidence_turn_ids": observation.get(
                        "evidence_turn_ids",
                        []
                    ),
                }
                for observation in trait_observations
            ],
        }

        trait_results.append(result)

    strength_order = {
        "strong": 0,
        "moderate": 1,
        "preliminary": 2,
    }

    trait_results.sort(
        key=lambda trait: (
            strength_order[
                trait["support_strength"]
            ],
            -trait["support"][
                "unique_sessions"
            ],
            -trait["support"][
                "observation_count"
            ],
            trait["title"],
        )
    )

    return trait_results, unmapped


# =========================================================
# MARKDOWN REPORT
# =========================================================

def build_markdown(
    trait_results,
    unmapped,
    observation_count,
    window_count,
):
    lines = []

    lines.append(
        "# Isaac Behavioral Trait Candidates — v0.1"
    )
    lines.append("")

    lines.append(
        "This report aggregates low-level behavioral observations from "
        "the pilot conversational sample. It is **not** yet the final "
        "Behavioral Persona Report."
    )
    lines.append("")

    lines.append("## Pilot Scope")
    lines.append("")
    lines.append(
        f"- Observations analyzed: **{observation_count}**"
    )
    lines.append(
        f"- Sample windows available: **{window_count}**"
    )
    lines.append(
        f"- Candidate traits with support: **{len(trait_results)}**"
    )
    lines.append(
        f"- Unmapped observations: **{len(unmapped)}**"
    )
    lines.append("")

    lines.append("## Support Labels")
    lines.append("")
    lines.append(
        "- **Strong:** at least 4 supporting observations, from at least "
        "3 sessions and 2 different years."
    )
    lines.append(
        "- **Moderate:** at least 2 observations from at least 2 sessions."
    )
    lines.append(
        "- **Preliminary:** everything else."
    )
    lines.append("")
    lines.append(
        "These labels describe evidence coverage in this pilot sample; "
        "they are not psychological certainty scores."
    )
    lines.append("")

    for trait in trait_results:
        support = trait["support"]

        lines.append(
            f"## {trait['title']}"
        )
        lines.append("")

        lines.append(
            f"**Support strength:** {trait['support_strength'].upper()}"
        )
        lines.append("")

        lines.append(trait["summary"])
        lines.append("")

        years = (
            ", ".join(
                str(year)
                for year in support[
                    "years_observed"
                ]
            )
            if support[
                "years_observed"
            ]
            else "unknown"
        )

        lines.append(
            f"- Supporting observations: "
            f"**{support['observation_count']}**"
        )

        lines.append(
            f"- Unique sessions: "
            f"**{support['unique_sessions']}**"
        )

        lines.append(
            f"- Unique windows: "
            f"**{support['unique_windows']}**"
        )

        lines.append(
            f"- Years represented: **{years}**"
        )

        lines.append(
            f"- Mean confidence weight: "
            f"**{support['mean_confidence_weight']:.2f} / 3.00**"
        )

        if trait["contexts"]:
            lines.append(
                "- Contexts observed: "
                + "; ".join(
                    trait["contexts"]
                )
            )

        lines.append("")
        lines.append("### Supporting observations")
        lines.append("")

        for observation in trait["observations"]:
            lines.append(
                f"- **{observation['behavior']}** "
                f"({observation['confidence']} confidence, "
                f"session {observation['session_id']}, "
                f"window {observation['window_id']}): "
                f"{observation['description']} "
                f"[turns: "
                + ", ".join(
                    str(turn_id)
                    for turn_id in observation[
                        "evidence_turn_ids"
                    ]
                )
                + "]"
            )

        lines.append("")

    if unmapped:
        lines.append("## Unmapped Behaviors")
        lines.append("")
        lines.append(
            "These observations were preserved but were not assigned "
            "to a broader candidate trait in the current mapping."
        )
        lines.append("")

        for observation in unmapped:
            lines.append(
                f"- `{observation.get('behavior')}` — "
                f"{observation.get('description')}"
            )

        lines.append("")

    lines.append("## Important Limitation")
    lines.append("")
    lines.append(
        "This V0.1 aggregation uses an explicit human-authored mapping "
        "from low-level behavior labels to broader candidate traits. "
        "It is intentionally transparent and easy to revise. Later "
        "versions can compare this approach with embedding-based or "
        "LLM-assisted semantic clustering."
    )
    lines.append("")

    return "\n".join(lines)


# =========================================================
# MAIN
# =========================================================

def main():
    observations = load_jsonl(
        OBSERVATIONS_PATH
    )

    windows = load_jsonl(
        WINDOWS_PATH
    )

    window_metadata = build_window_metadata(
        windows
    )

    trait_results, unmapped = aggregate(
        observations,
        window_metadata,
    )

    output = {
        "version": "0.1",
        "source_observations": str(
            OBSERVATIONS_PATH
        ),
        "source_windows": str(
            WINDOWS_PATH
        ),
        "observation_count": len(
            observations
        ),
        "window_count": len(
            windows
        ),
        "trait_count": len(
            trait_results
        ),
        "unmapped_observation_count": len(
            unmapped
        ),
        "traits": trait_results,
        "unmapped_observations": unmapped,
    }

    with JSON_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    markdown = build_markdown(
        trait_results=trait_results,
        unmapped=unmapped,
        observation_count=len(
            observations
        ),
        window_count=len(
            windows
        ),
    )

    MARKDOWN_OUTPUT_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print(
        f"Observations loaded:        "
        f"{len(observations):,}"
    )

    print(
        f"Sample windows loaded:      "
        f"{len(windows):,}"
    )

    print(
        f"Candidate traits produced:  "
        f"{len(trait_results):,}"
    )

    print(
        f"Unmapped observations:      "
        f"{len(unmapped):,}"
    )

    print()
    print("Trait summary:")
    print()

    for trait in trait_results:
        print(
            f"  [{trait['support_strength']:^11}] "
            f"{trait['title']:<38} "
            f"{trait['support']['observation_count']:>2} obs / "
            f"{trait['support']['unique_sessions']:>2} sessions"
        )

    print()
    print(
        f"JSON output:      "
        f"{JSON_OUTPUT_PATH}"
    )

    print(
        f"Readable report:  "
        f"{MARKDOWN_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
