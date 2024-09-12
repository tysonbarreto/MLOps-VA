from VA.logger import logger
from VA.exception import VAException
from VA.entity import Dataset
from VA.config import Config, MongoDBClient
from VA.components import DataIngestion
from VA.utils.utils import load_parquet, save_parquet, save_load_pickle
from VA.constants import *
import sys
from datetime import date
from scipy.stats import skew
from dataclasses import dataclass, field
from typing import List, Tuple, Any
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




@dataclass
class VAPredcitor:
    model_path: str
    df_path:str
    
    def __post_init__(self):
        models_df = save_load_pickle(filepath=BEST_PARAMS_DATASET_PATH, load=True)
        self.model = models_df['model'].iloc[0]
        self.df = load_parquet(RAW_DATASET_PATH)
        self.categorical_features = [feature for feature in self.df.columns if self.df[feature].dtype=='O']
        self.numerical_features = [feature for feature in self.df.columns if self.df[feature].dtype!='O']
        
    def predict(self):

        X = self.df.drop('case_status', axis=1)
        X['company_age'] = date.today().year - X['yr_of_estab']
        y = self.df['case_status']
        self.y = np.where(y=='Denied',1,0)
        
        self.or_columns = ['has_job_experience','requires_job_training','full_time_position','education_of_employee']
        self.oh_columns = ['continent','unit_of_wage','region_of_employment']
        self.transform_features = ['company_age', 'no_of_employees']
        
        X_processed = self.preprocessor().fit_transform(X)
        self.preds = self.model.predict(X_processed)
        return self.preds
    

    def calc_accuracy(self):
        # acc = accuracy_score(self.preds,self.y)
        f1 = f1_score(self.preds,self.y)
        # roc = roc_auc_score(self.preds,self.y)
        # precision = precision_score(self.preds,self.y)
        # recall = recall_score(self.preds,self.y)
        return round(f1*100,2)
    
    def preprocessor(self)->Pipeline:
        transform_pipline=Pipeline(steps=[('transformer', PowerTransformer(method='yeo-johnson'))])
        preprocessor = ColumnTransformer(
                        [
                            ('OneHotEncoder', OneHotEncoder(), self.oh_columns),
                            ('OrdinalEncoder', OrdinalEncoder() ,self.or_columns),
                            ('StandardScaler', StandardScaler(), self.numerical_features),
                            ('PowerTransformer', transform_pipline, self.transform_features )
                        ]
                    )
        return preprocessor