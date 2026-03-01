import os
from dataclasses import dataclass
from functools import lru_cache


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AppSettings:
    database_url: str | None
    server_url: str
    log_file: str
    log_level: str
    log_to_stdout: bool
    export_artifact_dir: str


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings(
        database_url=os.getenv("DATABASE_URL"),
        server_url=os.getenv("SERVER_URL", "http://localhost:8000"),
        log_file=os.getenv("LOG_FILE", "logs/log.txt"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_to_stdout=_as_bool(os.getenv("LOG_TO_STDOUT"), True),
        export_artifact_dir=os.getenv("EXPORT_ARTIFACT_DIR", os.path.join(os.getcwd(), "exports")),
    )


def get_database_url() -> str:
    database_url = get_settings().database_url
    if not database_url or not str(database_url).strip():
        raise RuntimeError("Missing required environment variable: DATABASE_URL")
    return database_url
