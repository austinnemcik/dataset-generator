from app.core.utils import extract_records, get_path_value, safe_json_dump


def normalize_alpaca_row(row: dict) -> dict | None:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    response = str(row.get("output", "")).strip()
    if input_text:
        instruction = f"{instruction}\n\nInput:\n{input_text}".strip()
    if instruction and response:
        return {"instruction": instruction, "response": response}
    return None


def normalize_with_field_mapper(row: dict, field_mapper: dict[str, str] | None) -> dict | None:
    if not field_mapper:
        return None
    instruction_key = field_mapper.get("instruction")
    response_key = field_mapper.get("response")
    if not instruction_key or not response_key:
        return None
    instruction = str(get_path_value(row, instruction_key) or "").strip()
    response = str(get_path_value(row, response_key) or "").strip()
    input_key = field_mapper.get("input")
    input_value = str(get_path_value(row, input_key) or "").strip() if input_key else ""
    if input_value:
        instruction = f"{instruction}\n\nInput:\n{input_value}".strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "response": response}


def normalize_chat_turns(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    pending_instruction: str | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(
            message.get("from", message.get("role", message.get("speaker", "")))
        ).strip().lower()
        content = str(
            message.get("value", message.get("content", message.get("text", "")))
        ).strip()
        if not content:
            continue
        if role in {"human", "user", "prompter"}:
            pending_instruction = content
        elif role in {"gpt", "assistant", "bot"} and pending_instruction:
            out.append({"instruction": pending_instruction, "response": content})
            pending_instruction = None
    return out


def detect_format(records: list, field_mapper: dict[str, str] | None = None) -> str:
    if field_mapper:
        return "mapped"
    sample = records[0] if records else None
    if isinstance(sample, dict):
        if {"instruction", "output"} <= set(sample.keys()):
            return "alpaca"
        if isinstance(sample.get("conversations"), list):
            return "sharegpt"
        if isinstance(sample.get("messages"), list):
            return "chatml"
    if isinstance(sample, list):
        return "chatml"
    return "unknown"


def normalize_import_records(
    records: list,
    *,
    detected_format: str,
    field_mapper: dict[str, str] | None = None,
) -> tuple[list[dict], int]:
    normalized: list[dict] = []
    invalid = 0
    for row in records:
        try:
            if detected_format == "alpaca" and isinstance(row, dict):
                parsed = normalize_alpaca_row(row)
                if parsed:
                    normalized.append(parsed)
                else:
                    invalid += 1
            elif detected_format == "mapped" and isinstance(row, dict):
                parsed = normalize_with_field_mapper(row, field_mapper)
                if parsed:
                    normalized.append(parsed)
                else:
                    invalid += 1
            elif detected_format == "sharegpt" and isinstance(row, dict):
                conversations = row.get("conversations", [])
                parsed_rows = normalize_chat_turns(conversations)
                if parsed_rows:
                    normalized.extend(parsed_rows)
                else:
                    invalid += 1
            elif detected_format == "chatml":
                messages = row if isinstance(row, list) else row.get("messages", []) if isinstance(row, dict) else []
                parsed_rows = normalize_chat_turns(messages)
                if parsed_rows:
                    normalized.extend(parsed_rows)
                else:
                    invalid += 1
            elif isinstance(row, dict):
                parsed = normalize_with_field_mapper(row, field_mapper)
                if parsed:
                    normalized.append(parsed)
                    continue
                if {"instruction", "response"} <= set(row.keys()):
                    instruction = str(row.get("instruction", "")).strip()
                    response = str(row.get("response", "")).strip()
                    if instruction and response:
                        normalized.append({"instruction": instruction, "response": response})
                    else:
                        invalid += 1
                else:
                    invalid += 1
            else:
                invalid += 1
        except Exception:
            invalid += 1
    return normalized, invalid


def normalize_scraper_text(text: str, response_char_limit: int) -> tuple[str, str] | None:
    normalized = " ".join(str(text or "").split())
    if len(normalized) < 8:
        return None
    snippet = normalized[:response_char_limit]
    instruction = "Summarize the scraped page content into a concise, accurate answer."
    response = snippet
    return instruction, response


def scraper_reference_card() -> dict:
    return {
        "title": "Scraper Intake Endpoint",
        "method": "POST",
        "endpoint": "/dataset/intake/scraper",
        "description": "Accepts normalized text payloads from external scrapers and imports them as training examples.",
        "curl": (
            "curl -X POST http://localhost:8000/dataset/intake/scraper\n"
            "  -H 'Content-Type: application/json'\n"
            "  -d '{\n"
            "    \"dataset_name\": \"scraped-support-pages\",\n"
            "    \"prompt\": \"Imported scraper text\",\n"
            "    \"records\": [\n"
            "      {\"text\": \"Reset your password from the account settings page...\", \"source_url\": \"https://example.com/help/reset-password\", \"title\": \"Password reset\"}\n"
            "    ]\n"
            "  }'"
        ),
        "payload_schema": {
            "records": [{"text": "string", "source_url": "string?", "title": "string?", "metadata": "object?"}],
            "dataset_name": "string?",
            "dataset_description": "string?",
            "model": "string?",
            "prompt": "string (default: Imported scraper text)",
            "dedupe_threshold": "float (0,1]",
            "dedupe_against_existing": "boolean (default: true)",
            "dedupe_within_payload": "boolean (default: true)",
            "max_records": "int (default: 2000)",
            "chunk_size": "int 1-500",
            "response_char_limit": "int 32-8000",
            "preview_only": "boolean",
            "preview_limit": "int 1-100",
        },
    }


def upload_reference_card() -> dict:
    return {
        "title": "File Intake Endpoint",
        "method": "POST",
        "endpoint": "/dataset/intake/upload",
        "description": "Accepts uploaded files and stores them either as training-example datasets or chunked source documents.",
        "curl": (
            "curl -X POST http://localhost:8000/dataset/intake/upload "
            "-F \"file=@./examples.jsonl\" "
            "-F \"intake_mode=examples\" "
            "-F \"dataset_name=uploaded-example-dataset\" "
            "-F \"prompt=Imported file dataset\""
        ),
        "form_fields": {
            "file": "binary file upload",
            "intake_mode": "examples | source_material | pretraining_data",
            "dataset_name": "string? (used for examples mode)",
            "dataset_description": "string? (used for examples mode)",
            "model": "string? (used for examples mode)",
            "prompt": "string (used for examples mode)",
            "dedupe_threshold": "float (0,1] used for examples mode",
            "dedupe_against_existing": "boolean",
            "dedupe_within_payload": "boolean",
            "chunk_char_size": "int > 0 used for source/pretraining modes",
            "chunk_overlap": "int >= 0 and < chunk_char_size",
        },
        "supported_files": {
            "examples": [".jsonl", ".json", ".csv"],
            "source_material": [".pdf", ".txt", ".csv", ".docx", ".md", ".json"],
            "pretraining_data": [".pdf", ".txt", ".csv", ".docx", ".md", ".json"],
        },
    }



