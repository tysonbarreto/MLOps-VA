from VA.logger import logger
from VA.exception import VAException
import yaml
import os
import sys
import dill
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Any

def save_load_pickle(filepath:str, data:Any=None, save:bool=False, load:bool=False):
    
    if save:
        with open(filepath, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        filename= os.path.split(filepath)[-1].split('.')[0]
        filepath = os.path.split(os.path.relpath(filepath))[0]
        logger.info(f"{filename} saved in {filepath} successfully")
        
    elif load:
        with open(filepath, 'rb') as handle:
            data = pickle.load(handle)
        filename= os.path.split(filepath)[-1].split('.')[0]
        filepath = os.path.split(os.path.relpath(filepath))[0]
        logger.info(f"{filename} loaded from {filepath} successfully")
        return data
    elif (save==True) & (load==True):
        logger.info(f"either save or load at one time")
        raise ValueError("either save or load at one time")

def save_parquet(filepath:str ,df:pd.DataFrame):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_parquet(filepath, index=False, engine='pyarrow')
        filename= os.path.split(filepath)[-1].split('.')[0]
        filepath = os.path.split(os.path.relpath(filepath))[0]
        logger.info(f"{filename} saved in {filepath} successfully")
    except Exception as e:
        logger.info(VAException(e, sys))

def load_parquet(filepath:str):
    try:
        df = pd.read_parquet(filepath)
        filename= os.path.split(filepath)[-1].split('.')[0]
        filepath = os.path.split(os.path.relpath(filepath))[0]
        logger.info(f"{filename} loaded from {filepath} successfully")
        return df
    except Exception as e:
        logger.info(VAException(e, sys))

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
    try:
        df = df.drop(columns=cols, axis=1)

        logger.info("Exited the drop_columns method of utils")
        
        return df
    except Exception as e:
        logger.info(VAException(e,sys))