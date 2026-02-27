import json
import re


def _strip_markdown_fences(text: str) -> str:
    content = text.strip()
    match = re.match(
        r"^```(?:json)?\s*(.*?)\s*```$", content, flags=re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return content


def _strip_common_artifacts(text: str) -> str:
    cleaned = text.lstrip("\ufeff").strip()
    cleaned = cleaned.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    cleaned = cleaned.replace("\ufeff", "")
    cleaned = re.sub(r"^\s*json\s*\n", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _extract_first_json_value(text: str) -> str:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text, idx=i)
            return text[i:end]
        except json.JSONDecodeError:
            continue
    return text


def _normalize_json_like_text(text: str) -> str:
    normalized = (
        text.replace("â€œ", '"')
        .replace("â€", '"')
        .replace("â€˜", "'")
        .replace("â€™", "'")
    )
    normalized = re.sub(r"/\*.*?\*/", "", normalized, flags=re.DOTALL)
    normalized = re.sub(r"^\s*//.*?$", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
    return normalized


def parse_json_with_fallback(raw: str):
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(text: str):
        if text not in seen:
            seen.add(text)
            candidates.append(text)

    base = _strip_common_artifacts(raw)
    no_fence = _strip_markdown_fences(base)
    extracted = _extract_first_json_value(no_fence)
    normalized = _normalize_json_like_text(extracted)

    add_candidate(raw)
    add_candidate(base)
    add_candidate(no_fence)
    add_candidate(extracted)
    add_candidate(normalized)

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_error = e

    if last_error:
        raise last_error
    return json.loads(raw)

