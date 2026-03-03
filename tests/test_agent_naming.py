from agent.naming import _normalize_topic_list, _parse_naming_metadata


def test_parse_naming_metadata_returns_clean_fields():
    parsed = _parse_naming_metadata('{"name":"  dataset-name ","description":" useful description "}')
    assert parsed == {"name": "dataset-name", "description": "useful description"}


def test_parse_naming_metadata_rejects_missing_fields():
    assert _parse_naming_metadata('{"name":"only-name"}') is None


def test_normalize_topic_list_dedupes_and_pads_with_fallback():
    parsed = {"topics": ["Async IO", "async io", "Event Loops"]}
    normalized = _normalize_topic_list(parsed, topic_count=4, fallback_topic="Python Concurrency")

    assert normalized == [
        "Async IO",
        "Event Loops",
        "Python Concurrency",
        "Python Concurrency",
    ]


def test_normalize_topic_list_falls_back_when_no_valid_topics():
    normalized = _normalize_topic_list({}, topic_count=2, fallback_topic="Base Topic")
    assert normalized == ["Base Topic"]
