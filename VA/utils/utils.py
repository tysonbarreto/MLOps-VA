from VA.logger import logger
from VA.exception import VAException
import yaml
import os
import sys
import dill
import numpy as np
import pandas as pd

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        logger.info(VAException(e, sys))
    


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        logger.info(VAException(e, sys))
        
def load_object(file_path: str) -> object:
    logger.info("Entered the load_object method of utils")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logger.info("Exited the load_object method of utils")

        return obj

    except Exception as e:
        logger.info(VAException(e, sys))
    


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        logger.info(VAException(e, sys))
        
        
    def drop_columns(df: pd.DataFrame, cols: list)-> pd.DataFrame:

        """
        drop the columns form a pandas DataFrame
        df: pandas DataFrame
        cols: list of columns to be dropped
        """
        logger.info("Entered drop_columns methon of utils")

        try:
            df = df.drop(columns=cols, axis=1)

            logger.info("Exited the drop_columns method of utils")
            
            return df
        except Exception as e:
            logger.info(VAException(e,sys))