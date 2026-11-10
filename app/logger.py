# ─────────────────────────────────────────────────────────────────────────────
# app/logger.py  —  Centralized logging for the entire application
#
# Sets up a logger that writes to BOTH the console and a log file.
# Import `logger` from this module anywhere you need to log.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import sys
from config import LOG_FILE

# Create the db/ directory if it doesn't exist yet
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "rag_system") -> logging.Logger:
    """
    Returns a configured logger instance.

    Args:
        name: Logger name (shown in log lines). Use __name__ in each module.

    Returns:
        A Python Logger object ready to use.
    """
    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers if the logger is already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Capture everything DEBUG and above

    # ── Formatter ─────────────────────────────────────────────────────────
    # Example line: 2024-01-15 10:30:00,123 | INFO     | rag_system | Message
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ────────────────────────────────────────────────────
    # Force UTF-8 on Windows so Unicode characters (arrows etc.) don't crash
    import io
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console_handler = logging.StreamHandler(utf8_stdout)
    console_handler.setLevel(logging.INFO)   # INFO and above go to console
    console_handler.setFormatter(fmt)

    # ── File handler ───────────────────────────────────────────────────────
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)     # DEBUG and above go to file
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Module-level logger — import this directly: `from app.logger import logger`
logger = get_logger("rag_system")
