import os
from datetime import date
from pathlib import Path
from from_root import from_root


DATABASE_NAME = "VA"

COLLECTION_NAME = "va_data"

MONGODB_URL_KEY = Path(os.path.abspath("notebook/KEY.env"))

RAW_DATASET_PATH = os.path.join(from_root(),Path("arifacts/raw/raw_dataset.parquet"))

TRAIN_SET_PATH = os.path.join(from_root(),Path("arifacts/processed/train_dataset.parquet"))

TEST_SET_PATH = os.path.join(from_root(),Path("arifacts/processed/test_dataset.parquet"))

TRANSFORMED_DATASET_PATH = os.path.join(from_root(),Path("arifacts/transformed/transformed_dataset.parquet"))

PARAMS_DATASET_PATH = os.path.join(from_root(),Path("arifacts/params/params_dataset.pkl"))
BEST_PARAMS_DATASET_PATH = os.path.join(from_root(),Path("arifacts/params/best_params.pkl"))

CONFIG_PATH = os.path.join(from_root(),Path("VA/config/config.yaml"))


