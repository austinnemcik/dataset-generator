from routes.data_processing import (
    detect_format,
    extract_records,
    normalize_import_records,
    normalize_scraper_text,
)


def test_extract_records_prefers_known_collection_keys():
    payload = {"data": [{"instruction": "i", "response": "r"}]}
    assert extract_records(payload) == [{"instruction": "i", "response": "r"}]


def test_detect_format_alpaca():
    fmt = detect_format([{"instruction": "inst", "output": "out"}])
    assert fmt == "alpaca"


def test_normalize_import_records_chatml():
    records = [
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        }
    ]
    normalized, invalid = normalize_import_records(records, detected_format="chatml")
    assert invalid == 0
    assert normalized == [{"instruction": "hello", "response": "world"}]


def test_normalize_scraper_text_enforces_min_length():
    assert normalize_scraper_text(" short ", 120) is None
    parsed = normalize_scraper_text("this is enough content", 10)
    assert parsed is not None
    assert parsed[1] == "this is en"
