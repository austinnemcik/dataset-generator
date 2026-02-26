import os
import logging
from logging.handlers import RotatingFileHandler

LOG_FILE = "logs/log.txt"
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
logger.addHandler(handler)


def saveToLog(message: str, logType: str = "INFO"):
    if logType == "INFO":
        logger.info(message)
    elif logType == "WARNING":
        logger.warning(message)
    elif logType == "ERROR":
        logger.error(message)
    return
