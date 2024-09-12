from VA import MongoDBClient, VAException, logger
import pandas as pd
from dataclasses import dataclass
import sys
import numpy as np

@dataclass
class Dataset:
    mongo_client: MongoDBClient
      
    def export_collection_as_dataframe(self)->pd.DataFrame:
        try:

            collection = self.mongo_client.collection

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise VAException(e,sys)
    