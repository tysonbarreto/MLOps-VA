from VA.logger import logger
from VA.exception import VAException
from VA.components import DataTransformation, DataIngestion, ModelEvaluation
from VA.entity import Dataset
from VA.config import MongoDBClient, Config
import sys

#-------------------------------PIPELINE-----------------------#


#DataFrame Loaded and saved locally

data_ingestion = DataIngestion(mongo_client=MongoDBClient(), params_config=Config())

df=data_ingestion.get_data()

train_set, test_set = data_ingestion.train_test_split_(df=df)

X_res, y_res = DataTransformation(train_set).apply_transformation()

model_eval = ModelEvaluation(X_res, y_res)

reports_df = model_eval.evaluate()

best_params_dict = model_eval.RGS_CV


