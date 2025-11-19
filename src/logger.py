import logging
import os
import sys
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Add console (stream) handler if not already present
_root_logger = logging.getLogger()
_has_stream = any(isinstance(h, logging.StreamHandler) for h in _root_logger.handlers)
if not _has_stream:
    _stream = logging.StreamHandler()
    _stream.setLevel(logging.INFO)
    _stream.setFormatter(logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"))
    _root_logger.addHandler(_stream)

# Global exception hook to log uncaught exceptions
def _log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    # Allow keyboard interrupt to behave normally
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.getLogger(__name__).error(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
    )

# Install excepthook once
if getattr(sys, "excepthook", None) is not _log_uncaught_exceptions:
    sys.excepthook = _log_uncaught_exceptions
