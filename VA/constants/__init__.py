import os
from datetime import date
from pathlib import Path


DATABASE_NAME = "VA"

COLLECTION_NAME = "va_data"

MONGODB_URL_KEY = Path(os.path.abspath("notebook/KEY.env"))

RAW_DATASET_PATH = os.path.abspath("arifacts/raw/raw_dataset.parquet")

TRAIN_SET_PATH = os.path.abspath("arifacts/processed/train_dataset.parquet")

TEST_SET_PATH = os.path.abspath("arifacts/processed/test_dataset.parquet")

TRANSFORMED_DATASET_PATH = os.path.abspath("arifacts/transformed/transformed_dataset.parquet")

PARAMS_DATASET_PATH = os.path.abspath("arifacts/params/params_dataset.pkl")
BEST_PARAMS_DATASET_PATH = os.path.abspath("arifacts/params/best_params.pkl")

CONFIG_PATH = os.path.abspath("VA/config/config.yaml")


