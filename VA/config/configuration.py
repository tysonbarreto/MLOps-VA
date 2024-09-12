from VA.constants import *
from VA.utils import read_yaml_file
from VA.exception import VAException
from VA.logger import logger

from typing import OrderedDict, Any
from dataclasses import dataclass, field
from dotenv import dotenv_values
import pymongo
import pandas as pd
from pathlib import Path
import sys
from box import ConfigBox

import certifi

ca = certifi.where()


@dataclass
class MongoDBClient:
    mongo_db_key: str = MONGODB_URL_KEY
    collection_name: str = COLLECTION_NAME
    database_name: str = DATABASE_NAME
    
    def __post_init__(self)->None:
        try:
            env_ = dotenv_values(self.mongo_db_key)
            cnxn_string = env_.values()
            self.client = pymongo.MongoClient(cnxn_string,  tlsCAFile=ca)
            self.database = self.client[self.database_name]
            self.collection = self.database[self.collection_name]
            logger.info("Connection with MongoDB established!")
        except Exception as e:
            logger.info(VAException(e, sys))

@dataclass
class Config:
    config: str = CONFIG_PATH
    
    def __post_init__(self):
        self.params_dict = read_yaml_file(self.config)
        self.params = ConfigBox(self.params_dict)
        self.test_size = self.params.split.test_size
    