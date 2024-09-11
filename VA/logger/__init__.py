import logging
import os
from pathlib import Path
import sys

from from_root import from_root
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_dir = 'logs'

logs_path = Path(os.path.join(from_root(), log_dir, "logs.log"))

os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    #filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(logs_path)]
)

logger = logging.getLogger("VALogger")