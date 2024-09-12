import pymongo
import pandas as pd
import os
from pathlib import Path
from dotenv import dotenv_values
from VA.logger import logger
from VA.entity import Dataset
from VA.exception import VAException
from VA.utils import drop_columns, save_parquet 
from VA.config import MongoDBClient, Config
from VA.constants import *
from dataclasses import dataclass
import sys
from typing import Tuple
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestion:
    mongo_client:MongoDBClient
    params_config:Config
    
    def __post_init__(self):
        client = self.mongo_client
        self.df = Dataset(mongo_client=client).export_collection_as_dataframe()
        self.test_size = self.params_config.test_size

    def get_data(self, path:Path = RAW_DATASET_PATH):
        save_parquet(filepath=path, df=self.df)
        logger.info("Raw df is loaded successfully!")
        return self.df

    def train_test_split_(self, df:pd.DataFrame)->Tuple[pd.DataFrame, pd.DataFrame]:
        train_set, test_set = train_test_split(df, test_size=self.test_size)
        save_parquet(filepath=TRAIN_SET_PATH, df=train_set)
        save_parquet(filepath=TEST_SET_PATH, df=test_set)
        logger.info("train set and test set are loaded successfully!")
        
        return train_set, test_set

    

    
    