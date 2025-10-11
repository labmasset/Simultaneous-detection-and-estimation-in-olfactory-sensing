import logging
import sys
from pathlib import Path
from typing import Union
from datetime import datetime

# TODO: logger level does not correctly passed in from config
def setup_logging(
    level: str = "INFO", log_file: Union[str, Path, None] = None
) -> logging.Logger:
    """Configure the root logger with console and file handlers."""
    logger = logging.getLogger()
    if logger.handlers:
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "log"
    log_dir.mkdir(exist_ok=True)

    if log_file is None:
        prog = Path(sys.argv[0]).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{prog}_{ts}.log"
    else:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = log_dir / log_path.name

    file_handler = logging.FileHandler(str(log_path))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Logging to %s", log_path)

    return logger
