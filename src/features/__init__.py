import logging
import os
import sys
from pathlib import Path

from features.logger_custom import get_logger

src = Path(os.getcwd())
sys.path.append(src.joinpath("features").as_posix())
sys.path.append(src.joinpath("lib").as_posix())

__all__ = ["get_logger"]

get_logger(
    logger_name="preprocessing", log_level=logging.DEBUG, log_file="data/logger.log"
)
