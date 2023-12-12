from dotenv import load_dotenv
import json
import os
import typing as tp

class ADDMSLIM:
    def __init__(self) -> None:
        self.popular = [10440, 15297, 9728, 13865, 4151, 3734, 2657, 4880, 142, 6809]
        load_dotenv(".env")
        ADDMSLIM_DIR = os.getenv("ADMMSLIM_DIR")
        if ADDMSLIM_DIR is None:
            raise ValueError("ADDMSLIM_DIR environment variable not found")
        file_path = os.path.join(os.getcwd(), ADDMSLIM_DIR)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        with open(file_path, "r", encoding="utf-8") as f:
            self.recommendations = json.load(f)

    def recommend(self, user_id: int) -> tp.List[int]:
        if str(user_id) in self.recommendations:
            return self.recommendations[str(user_id)]
        else:
            return self.popular