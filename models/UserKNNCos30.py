import os
import pickle

import pandas as pd
from dotenv import load_dotenv
from rectools import Columns
from rectools.dataset import Dataset, Interactions

from scripts.userknn import UserKnn

load_dotenv(".env")
USERKNNCOS30_NAME = os.getenv("USERKNNCOS30_NAME")
DATASET_DIR = os.getenv("DATASET_DIR")


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print(module, name)
        if module == "userknn" and name == "UserKnn":
            return UserKnn
        return super().find_class(module, name)


class UserKnnCos30:
    def __init__(self):
        self.model = self.load_model()
        self.data_for_predict = self.load_data_for_predict()

    def load_model(self):
        if os.path.exists(USERKNNCOS30_NAME):
            with open(USERKNNCOS30_NAME, "rb") as file:
                return CustomUnpickler(file).load()
        return None

    def load_data_for_predict(self):
        if self.model is not None:
            interactions_df = pd.read_csv(DATASET_DIR, parse_dates=["last_watch_dt"]).rename(
                columns={"last_watch_dt": Columns.Datetime, "total_dur": Columns.Weight}
            )
            interactions = Interactions(interactions_df).df
            data_for_predict = Dataset.construct(
                interactions,
            )
            return data_for_predict
        else:
            return None

    def predict(self, user_id):
        if self.data_for_predict is not None:
            return self.model.recommend_items_to_user(dataset=self.data_for_predict, user_id=user_id)
        else:
            return "No prediction data available"
