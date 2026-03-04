import csv
import io
import json
import re
from pathlib import Path


TEXT_LIKE_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cpp",
    ".cs",
    ".go",
    ".rs",
    ".html",
    ".css",
    ".scss",
    ".sql",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".xml",
    ".sh",
    ".ps1",
}


def safe_json_dump(value) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def get_path_value(payload, path: str | None):
    if not path:
        return None
    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current.get(part)
            continue
        if isinstance(current, list):
            try:
                index = int(part)
            except (TypeError, ValueError):
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def extract_records(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "rows", "examples", "items", "records", "messages"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise ValueError("Unsupported import payload shape.")


def parse_embedding(embedding_raw: str | None) -> list[float] | None:
    if not embedding_raw:
        return None
    try:
        parsed = json.loads(embedding_raw)
        if not isinstance(parsed, list) or not parsed:
            return None
        return [float(x) for x in parsed]
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def sse_message(*, data: dict, event: str | None = None, event_id: str | None = None) -> str:
    lines: list[str] = []
    if event_id:
        lines.append(f"id: {event_id}")
    if event:
        lines.append(f"event: {event}")
    payload = json.dumps(data, separators=(",", ":"))
    for line in payload.splitlines() or ["{}"]:
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def parse_last_event_index(last_event_id: str | None) -> int:
    if not last_event_id:
        return -1
    if not last_event_id.startswith("item:"):
        return -1
    parts = last_event_id.split(":")
    if len(parts) < 2:
        return -1
    try:
        return int(parts[1])
    except (TypeError, ValueError):
        return -1


def _decode_text(contents: bytes, filename: str) -> str:
    try:
        return contents.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Unable to decode {filename} as UTF-8 text.") from exc


def _normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _strip_markdown_syntax(text: str) -> str:
    text = re.sub(r"```.*?```", lambda m: m.group(0).strip("`"), text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_~`>#]", "", text)
    return _normalize_whitespace(text)


def _flatten_json_to_text(value, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_json_to_text(item, next_prefix))
        return lines
    if isinstance(value, list):
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            lines.extend(_flatten_json_to_text(item, next_prefix))
        return lines
    rendered = str(value).strip()
    if not rendered:
        return lines
    if prefix:
        lines.append(f"{prefix}: {rendered}")
    else:
        lines.append(rendered)
    return lines


def _remove_page_artifacts(page_texts: list[str]) -> list[str]:
    page_lines = [[line.strip() for line in text.splitlines() if line.strip()] for text in page_texts]
    if len(page_lines) < 2:
        return ["\n".join(lines) for lines in page_lines]

    first_counts: dict[str, int] = {}
    last_counts: dict[str, int] = {}
    for lines in page_lines:
        if lines:
            first_counts[lines[0]] = first_counts.get(lines[0], 0) + 1
            last_counts[lines[-1]] = last_counts.get(lines[-1], 0) + 1

    cleaned_pages: list[str] = []
    for lines in page_lines:
        cleaned = list(lines)
        if cleaned and first_counts.get(cleaned[0], 0) > 1:
            cleaned = cleaned[1:]
        if cleaned and last_counts.get(cleaned[-1], 0) > 1:
            cleaned = cleaned[:-1]
        cleaned = [
            line
            for line in cleaned
            if not re.fullmatch(r"(page\s+)?\d+(\s+of\s+\d+)?", line.strip(), flags=re.IGNORECASE)
        ]
        cleaned_pages.append("\n".join(cleaned))
    return cleaned_pages


def _extract_pdf_text(contents: bytes, filename: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("PDF intake requires pypdf to be installed.") from exc

    try:
        reader = PdfReader(io.BytesIO(contents))
    except Exception as exc:
        raise ValueError(f"Unable to parse PDF file: {filename}") from exc

    pages = [page.extract_text() or "" for page in reader.pages]
    cleaned_pages = _remove_page_artifacts(pages)
    return _normalize_whitespace("\n\n".join(cleaned_pages))


def _extract_docx_text(contents: bytes, filename: str) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise ValueError("DOCX intake requires python-docx to be installed.") from exc

    try:
        document = Document(io.BytesIO(contents))
    except Exception as exc:
        raise ValueError(f"Unable to parse DOCX file: {filename}") from exc

    text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    return _normalize_whitespace(text)


def _csv_rows_to_text(text: str) -> str:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV file must include a header row.")

    rows: list[str] = []
    for index, row in enumerate(reader, start=1):
        columns = [f"{key}: {value}" for key, value in row.items() if str(value or "").strip()]
        if columns:
            rows.append(f"Row {index}\n" + "\n".join(columns))
    if not rows:
        raise ValueError("CSV file did not contain any data rows.")
    return "\n\n".join(rows)


def parse_file_for_examples(contents: bytes, filename: str):
    suffix = Path(filename).suffix.lower()

    if suffix == ".jsonl":
        raw_rows: list[dict] = []
        text = _decode_text(contents, filename)
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL on line {line_number}.") from exc
            raw_rows.append(parsed)
        if not raw_rows:
            raise ValueError("JSONL file did not contain any records.")
        return raw_rows

    if suffix == ".json":
        try:
            parsed = json.loads(_decode_text(contents, filename))
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed JSON file.") from exc
        records = extract_records(parsed)
        if not records:
            raise ValueError("JSON file did not contain any records.")
        return records

    if suffix == ".csv":
        text = _decode_text(contents, filename)
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValueError("CSV file must include a header row.")
        rows = [row for row in reader]
        if not rows:
            raise ValueError("CSV file did not contain any rows.")
        return rows

    raise ValueError(f"Unsupported example file type: {suffix or filename}")


def parse_file_for_material(contents: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        text = _extract_pdf_text(contents, filename)
    elif suffix == ".docx":
        text = _extract_docx_text(contents, filename)
    elif suffix == ".md":
        text = _strip_markdown_syntax(_decode_text(contents, filename))
    elif suffix == ".json":
        try:
            parsed = json.loads(_decode_text(contents, filename))
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed JSON file.") from exc
        text = "\n".join(_flatten_json_to_text(parsed))
    elif suffix == ".jsonl":
        text_rows = []
        for row in parse_file_for_examples(contents, filename):
            text_rows.extend(_flatten_json_to_text(row))
        text = "\n".join(text_rows)
    elif suffix == ".csv":
        text = _csv_rows_to_text(_decode_text(contents, filename))
    elif suffix in TEXT_LIKE_EXTENSIONS or not suffix:
        text = _normalize_whitespace(_decode_text(contents, filename))
    else:
        raise ValueError(f"Unsupported source material file type: {suffix or filename}")

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("No usable text could be extracted from uploaded file.")
    return cleaned


def chunk_text(text: str, *, chunk_char_size: int = 2000, chunk_overlap: int = 200) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if chunk_char_size <= 0:
        raise ValueError("chunk_char_size must be greater than 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_char_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_char_size.")

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_char_size)
        if end < len(cleaned):
            boundary = cleaned.rfind("\n", start, end)
            if boundary == -1:
                boundary = cleaned.rfind(" ", start, end)
            if boundary > start + int(chunk_char_size * 0.6):
                end = boundary
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(0, end - chunk_overlap)
    return chunks

