from agent.grading import (
    _build_rejected_payload,
    _empty_grading_result,
    _evaluate_loaded_rows,
    _normalize_category,
)


def test_normalize_category_accepts_known_values():
    assert _normalize_category(" Technology_Software ") == "technology_software"
    assert _normalize_category("unknown") is None


def test_empty_grading_result_has_expected_shape():
    result = _empty_grading_result("no examples")
    assert result == {
        "accepted_examples": [],
        "dataset_score": 0.0,
        "notes": "no examples",
        "category": None,
    }


def test_evaluate_loaded_rows_accepts_and_rejects_correctly():
    examples = [
        {"instruction": "good", "response": "x" * 50},
        {"instruction": "bad", "response": "short"},
    ]
    loaded = {
        "rows": [
            {"idx": 0, "score_0_10": 9.2, "accept": 1, "reason": "good"},
            {"idx": 1, "score_0_10": 9.9, "accept": 1, "reason": "too short in practice"},
        ],
        "dataset_score_0_10": 8.7,
        "notes": "solid overall",
        "category": "technology_software",
    }

    accepted, rejected, dataset_score, notes, category = _evaluate_loaded_rows(examples, loaded)

    assert accepted == [examples[0]]
    assert len(rejected) == 1
    assert rejected[0]["idx"] == 1
    assert rejected[0]["reason"] == "too short in practice"
    assert dataset_score == 8.7
    assert notes == "solid overall"
    assert category == "technology_software"


def test_evaluate_loaded_rows_voice_alignment_does_not_reject_short_character_reply():
    examples = [
        {"instruction": "stay in character", "response": "Fine. Let's go."},
    ]
    loaded = {
        "rows": [
            {"idx": 0, "score_0_10": 8.8, "accept": 1, "reason": "in-character"},
        ],
        "dataset_score_0_10": 8.8,
        "notes": "voice is consistent",
        "category": "creative_media",
    }

    accepted, rejected, dataset_score, notes, category = _evaluate_loaded_rows(
        examples,
        loaded,
        grading_lens="voice_alignment",
    )

    assert accepted == examples
    assert rejected == []
    assert dataset_score == 8.8
    assert notes == "voice is consistent"
    assert category == "creative_media"


def test_build_rejected_payload_preserves_failure_context():
    rejected = [
        {
            "idx": 3,
            "example": {"instruction": "inst", "response": "resp"},
            "score": 4.5,
            "reason": "low_score",
        }
    ]

    assert _build_rejected_payload(rejected) == [
        {
            "original_idx": 3,
            "instruction": "inst",
            "response": "resp",
            "failure_reason": "low_score",
            "failure_score": 4.5,
        }
    ]
