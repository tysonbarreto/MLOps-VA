from VA.logger import logger
from VA.exception import VAException
import sys

#logger.info("Welcome to logger")


try:
    2/0
except Exception as e:
    logger.info(VAException(e,sys))
