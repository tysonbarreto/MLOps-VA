from VA.logger import logger
from VA.exception import VAException
from VA.entity import Dataset
from VA.config import Config, MongoDBClient
from VA.components import DataIngestion
from VA.utils import load_parquet, save_parquet, save_load_pickle
from VA.constants import *
import sys
from datetime import date
from scipy.stats import skew
from dataclasses import dataclass, field
from typing import List, Tuple
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, precision_score, classification_report, ConfusionMatrixDisplay, recall_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.combine import SMOTETomek,SMOTEENN
from copy import deepcopy as dc


def calc_accuracy(actual, preds):
    acc = accuracy_score(actual,preds)
    f1 = f1_score(actual,preds)
    roc = roc_auc_score(actual,preds)
    precision = precision_score(actual,preds)
    recall = recall_score(actual,preds)
    return acc, f1, roc, precision, recall

models = {
        "RandomForest":RandomForestClassifier(),
        "GradientBoosting":GradientBoostingClassifier(),
    }

@dataclass
class ModelEvaluation:
    X_res: np.array
    y_res: np.array
    models:dict=field(default_factory=lambda: models)
    
    def evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_res,self.y_res, test_size=0.2, random_state=42)
        
        models_list = []
        
        for model_name, model in tqdm(self.models.items()):
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            acc, f1, roc, precision, recall = calc_accuracy(y_test, y_test_pred)
            
            models_list.append({
                'model_name':model_name,"model":model,"acc":acc, "f1":f1, "roc":roc, "precision":precision, "recall":recall
            })
            
            print(f"{model_name:>20}\n {'*'*30}\n- Accuray: {acc:.4f}\n- F1 score: {f1:.4f}\n- Roc Auc: {roc:.4f}\n- Precision:{precision:.4f}\n- Recall:{recall:.4f}\n {'*'*30}\n {'*'*30}")

        self.report_df = pd.DataFrame(models_list)
        save_load_pickle(filepath=PARAMS_DATASET_PATH,data=self.report_df, save=True)
        return self.report_df
    
    @property
    def RGS_CV(self, params:dict=None)->dict:
        
        if params==None:
            logger.info("default ranges utilized for grid search as no params dict was provided")
            model_params={
                "RandomForest":{
                    "max_depth":[10,12,None,15,20],
                    "max_features": ['sqrt', 'log2', None],
                    "n_estimators": [10,50,100,200]
                },
                "GradientBoosting":{
                    "max_depth":[10,12,None,15,20],
                    "max_features": ['sqrt', 'log2', None],
                    "n_estimators": [10,50,100,200]
                }
            }
        best_params = {}
        logger.info("Randomized GS is being processed......")
        for index,row in tqdm(self.report_df.iterrows()):
            random = RandomizedSearchCV(
                estimator=row.model,
                param_distributions=model_params.get(row.model_name),
                n_iter=100,
                cv=3,
                verbose=2,
                n_jobs=-1
            )
            random.fit(self.X_res, self.y_res)
            best_params.update(
                {
                    row.model_name:random.best_params_
                }
            )
            save_load_pickle(filepath=BEST_PARAMS_DATASET_PATH,data=self.report_df, save=True)
            return best_params