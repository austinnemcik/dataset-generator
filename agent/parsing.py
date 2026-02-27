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


def _has_unbalanced_structure(text: str) -> bool:
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                return True
            top = stack.pop()
            if (top == "[" and ch != "]") or (top == "{" and ch != "}"):
                return True
    return in_string or bool(stack)


def _looks_truncated_json(text: str) -> bool:
    trimmed = text.strip()
    if not trimmed:
        return True
    starts_json = trimmed[0] in "[{"
    ends_json = trimmed[-1] in "]}"
    if starts_json and not ends_json:
        return True
    return _has_unbalanced_structure(trimmed)


def parse_json_with_fallback(raw: str, *, require_top_level_list: bool = False):
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(text: str):
        if text not in seen:
            seen.add(text)
            candidates.append(text)

    base = _strip_common_artifacts(raw)
    if require_top_level_list and _looks_truncated_json(base):
        raise json.JSONDecodeError("Likely truncated or unbalanced JSON output", base, 0)
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
            parsed = json.loads(candidate)
            if require_top_level_list and not isinstance(parsed, list):
                continue
            return parsed
        except json.JSONDecodeError as e:
            last_error = e

    if last_error:
        raise last_error
    parsed = json.loads(raw)
    if require_top_level_list and not isinstance(parsed, list):
        raise json.JSONDecodeError("Top-level JSON must be an array", raw, 0)
    return parsed

