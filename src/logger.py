import logging
import os
from datetime import datetime
import warnings 


# Log file name and directory setup
log_file_name = f"{datetime.now().strftime('%m%d%y__%H%M%S')}.log"
log_directory = os.path.join(os.getcwd(), "logs")
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, log_file_name)

# Basic logging configuration
logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Set this to INFO to capture only necessary info-level logs and above
)


logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)
