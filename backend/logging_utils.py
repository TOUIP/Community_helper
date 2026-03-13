import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FALLBACK_LOG_DIR = BASE_DIR / "logs"
STANDARD_LOG_DIR = Path("/var/log/community_helper")
_configured_loggers: dict[str, logging.Logger] = {}


def resolve_log_dir() -> Path:
    configured = os.getenv("APP_LOG_DIR", "").strip()
    candidates = [Path(configured)] if configured else [STANDARD_LOG_DIR, FALLBACK_LOG_DIR]

    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.touch(exist_ok=True)
            probe.unlink(missing_ok=True)
            return path
        except OSError:
            continue

    FALLBACK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return FALLBACK_LOG_DIR


def get_logger(name: str, filename: str) -> logging.Logger:
    if name in _configured_loggers:
        return _configured_loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_dir = resolve_log_dir()
    file_handler = RotatingFileHandler(
        log_dir / filename,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    _configured_loggers[name] = logger
    return logger


def get_runs_log_path() -> Path:
    log_dir = resolve_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "runs.jsonl"
