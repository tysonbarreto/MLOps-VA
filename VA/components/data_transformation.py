from VA.logger import logger
from VA.exception import VAException
from VA.entity import Dataset
from VA.config import Config, MongoDBClient
from VA.components import DataIngestion
from VA.utils import load_parquet
import sys
from datetime import date
from scipy.stats import skew
from dataclasses import dataclass
from typing import List, Tuple

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
class DataTransformation:
    df:pd.DataFrame
    
    def __post_init__(self):
        self.categorical_features = [feature for feature in self.df.columns if self.df[feature].dtype=='O']
        self.numerical_features = [feature for feature in self.df.columns if self.df[feature].dtype!='O']
        
    def apply_transformation(self)->Tuple[np.array, np.array]:
        
        X = self.df.drop('case_status', axis=1)
        y = self.df['case_status']
        y = np.where(y=='Denied',1,0)
        X['company_age'] = date.today().year - X['yr_of_estab']
        
        self.or_columns = ['has_job_experience','requires_job_training','full_time_position','education_of_employee']
        self.oh_columns = ['continent','unit_of_wage','region_of_employment']
        self.transform_features = ['company_age', 'no_of_employees']
        
        X_processed = self.preprocessor().fit_transform(X)
        
        logger.info(f"preprocessor has been loaded and applied to the dataset successfully")
        
        smt = SMOTEENN(random_state=42, sampling_strategy='minority')
        X_res, y_res = smt.fit_resample(X_processed,y)
        
        return X_res, y_res
            

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
    
    
              
