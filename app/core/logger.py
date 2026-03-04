import logging
import os
from logging.handlers import RotatingFileHandler

from app.core.config import PROJECT_ROOT, get_settings

settings = get_settings()
LOG_FILE = settings.log_file
LOG_LEVEL = settings.log_level
LOG_TO_STDOUT = settings.log_to_stdout

os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)

logger = logging.getLogger("dataset_generator")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = RotatingFileHandler(LOG_FILE, encoding="utf-8", maxBytes=1_048_576, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if LOG_TO_STDOUT:
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)


def saveToLog(message: str, logType: str = "INFO"):
    level_name = (logType or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.log(level, message)


def log_event(event: str, logType: str = "INFO", **fields):
    parts = [f"[{event}]"]
    for key in sorted(fields):
        value = fields[key]
        parts.append(f"{key}={value!r}")
    saveToLog(" ".join(parts), logType)

