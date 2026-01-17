import logging
import os 
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join("logs", LOG_FILE_NAME)
os.makedirs(LOG_FILE_PATH, exist_ok=True)

LOG_FILE = os.path.join(LOG_FILE_PATH, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE,
    format="[%(asctime)s] %(lineno)d %(name)s -%(levelname)s %(message)s",
    level=logging.INFO,
)

