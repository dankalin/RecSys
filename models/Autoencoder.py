import os
import pickle

import pandas as pd
from dotenv import load_dotenv
from rectools import Columns
from rectools.dataset import Dataset, Interactions
import json
import typing as tp

load_dotenv(".env")
AUTOENCODER_DIR = os.getenv("AUTOENCODER_DIR")


class AutoEncoder:
    def __init__(self) -> None:
        self.popular = [10440, 15297, 9728, 13865, 4151, 3734, 2657, 4880, 142, 6809]
        file_path = os.path.join(os.getcwd(), AUTOENCODER_DIR)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                self.recommendations = json.load(f)["item_id"]
        else: None
    def recommend(self, user_id: int) -> tp.List[int]:
        if str(user_id) in self.recommendations:
            return self.recommendations[str(user_id)]
        else:
            return self.popular