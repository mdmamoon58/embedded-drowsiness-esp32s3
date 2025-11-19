import sys
from typing import Any, Tuple


def error_message_detail(error: Any, error_detail: Any = None) -> str:
    """Build a readable error message with filename and line number.

    Accepts either the sys module (with exc_info) or a raw exc_info tuple
    for error_detail. Falls back gracefully when not in an exception context.
    """
    if error_detail is None:
        error_detail = sys

    exc_type = None
    exc_value = None
    exc_tb = None

    try:
        if hasattr(error_detail, "exc_info"):
            exc_info: Tuple[Any, Any, Any] = error_detail.exc_info()
        elif isinstance(error_detail, tuple) and len(error_detail) == 3:
            exc_info = error_detail  # type: ignore[assignment]
        else:
            exc_info = (None, None, None)

        exc_type, exc_value, exc_tb = exc_info
    except Exception:
        exc_tb = None

    try:
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "<unknown>"
        line_no = exc_tb.tb_lineno if exc_tb else -1
    except Exception:
        file_name = "<unknown>"
        line_no = -1

    message_text = str(error)
    return f"Error in [{file_name}] at line [{line_no}]: {message_text}"


class CustomException(Exception):
    """Custom exception that formats and logs contextual error information."""

    def __init__(self, error: Any, error_detail: Any = None) -> None:
        message = error_message_detail(error, error_detail if error_detail is not None else sys)
        super().__init__(message)
        self.message = message

        # Best-effort logging without introducing import cycles
        try:
            from src.logger import logging  # defer import
            logging.getLogger(__name__).error(message)
        except Exception:
            # If logging isn't available yet, skip silently
            pass

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}({self.message!r})"