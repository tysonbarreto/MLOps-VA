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
    model: Any
    
    model
    