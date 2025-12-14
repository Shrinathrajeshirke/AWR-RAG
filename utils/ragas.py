import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Generate log filename with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure main logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

logging.info("Logging initialized.")
logging.info(f"Log file: {LOG_FILE_PATH}")

# Create separate logger for RAGAS evaluation
ragas_logger = logging.getLogger('ragas_evaluation')
ragas_logger.setLevel(logging.INFO)

# Create RAGAS log file with date (one file per day)
RAGAS_LOG_FILE = f"ragas_eval_{datetime.now().strftime('%Y%m%d')}.log"
RAGAS_LOG_PATH = os.path.join(LOG_DIR, RAGAS_LOG_FILE)

ragas_handler = logging.FileHandler(RAGAS_LOG_PATH)
ragas_handler.setFormatter(logging.Formatter(
    '%(asctime)s - RAGAS - %(levelname)s - %(message)s'
))
ragas_logger.addHandler(ragas_handler)

logging.info(f"RAGAS logger initialized. Log file: {RAGAS_LOG_PATH}")