import logging
import os

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if logger already exists
    if not logger.hasHandlers():
        logger.addHandler(handler)

    return logger
