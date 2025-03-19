import logging
import os
from datetime import datetime

# Suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Log directory and file setup
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, log_file_name)

# Basic logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode='a',  # Append mode to avoid overwriting
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO  # Capture INFO and above
)

# Reduce verbosity of external libraries
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

# Optional: Log to console as well
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

# Confirmation
logging.info("Logger initialized successfully.")
