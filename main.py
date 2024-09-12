from VA.logger import logger
from VA.exception import VAException
from VA.components import DataTransformation, DataIngestion, ModelEvaluation, VAPredcitor
from VA.constants import RAW_DATASET_PATH, BEST_PARAMS_DATASET_PATH, TEST_SET_PATH
from VA.utils import load_parquet
from VA.entity import Dataset
from VA.config import MongoDBClient, Config
import sys
from from_root import from_root
from pathlib import Path
import os

#-------------------------------PIPELINE-----------------------#


#DataFrame Loaded and saved locally

data_ingestion = DataIngestion(mongo_client=MongoDBClient(), params_config=Config())
df=data_ingestion.get_data()
train_set, test_set = data_ingestion.train_test_split_(df=df)
X_res, y_res = DataTransformation(train_set).apply_transformation()
model_eval = ModelEvaluation(X_res, y_res)
reports_df = model_eval.evaluate()
# best_params_dict = model_eval.RGS_CV

perdictor = VAPredcitor(model_path=RAW_DATASET_PATH, df_path=BEST_PARAMS_DATASET_PATH)
preds = perdictor.predict()
print(perdictor.calc_accuracy())

