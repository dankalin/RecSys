from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse
from implicit.nearest_neighbours import ItemItemRecommender
from rectools.dataset import Dataset
from typing import Dict

class UserKnn:
    """Класс для обучения и предсказания модели UserKNN, основанной на модели ItemKNN из implicit.nearest_neighbours"""

    def __init__(self, model: ItemItemRecommender, num_users: int = 50):
        self.num_users = num_users
        self.model = model
        self.is_fitted = False
        self.users_mapping = {}
        self.items_mapping = {}
        self.interaction_matrix = None
        self.watched = None
        self.item_idf = None

    def get_mappings(self, train):
        self.users_inv_mapping = dict(enumerate(train["user_id"].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(enumerate(train["item_id"].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    def get_matrix(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        weight_col: str = None,
        users_mapping: Dict[int, int] = None,
        items_mapping: Dict[int, int] = None,
    ):
        if weight_col:
            weights = df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)

        self.interaction_matrix = sp.sparse.coo_matrix(
            (weights, (df[item_col].map(self.items_mapping.get), df[user_col].map(self.users_mapping.get)))
        )

        self.watched = (
            df.groupby(user_col, as_index=False).agg({item_col: list}).rename(columns={user_col: "sim_user_id"})
        )

        return self.interaction_matrix

    def idf(self, n: int, x: float):
        return np.log((1 + n) / (1 + x) + 1)

    def _count_item_idf(self, df: pd.DataFrame):
        item_cnt = Counter(df["item_id"].values)
        item_idf = pd.DataFrame.from_dict(item_cnt, orient="index", columns=["doc_freq"]).reset_index()
        item_idf["idf"] = item_idf["doc_freq"].apply(lambda x: self.idf(self.n, x))
        self.item_idf = item_idf

    def fit(self, train: pd.DataFrame):
        self.user_knn = self.model
        self.get_mappings(train)
        self.weights_matrix = self.get_matrix(train, users_mapping=self.users_mapping, items_mapping=self.items_mapping)

        self.n = train.shape[0]
        self._count_item_idf(train)

        self.user_knn.fit(self.weights_matrix)
        self.is_fitted = True

    def _generate_recs_mapper(
        self, model: ItemItemRecommender, user_mapping: Dict[int, int], user_inv_mapping: Dict[int, int], N: int
    ):
        def _recs_mapper(user):
            user_id = self.users_mapping[user]
            users, sim = model.similar_items(user_id, N=N)
            return [self.users_inv_mapping[user] for user in users], sim

        return _recs_mapper

    def predict(self, test: pd.DataFrame, N_recs: int = 10):
        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        mapper = self._generate_recs_mapper(
            model=self.user_knn,
            user_mapping=self.users_mapping,
            user_inv_mapping=self.users_inv_mapping,
            N=self.N_users,
        )

        recs = pd.DataFrame({"user_id": test["user_id"].unique()})
        recs["sim_user_id"], recs["sim"] = zip(*recs["user_id"].map(mapper))
        recs = recs.set_index("user_id").apply(pd.Series.explode).reset_index()

        recs = (
            recs[~(recs["user_id"] == recs["sim_user_id"])]
            .merge(self.watched, on=["sim_user_id"], how="left")
            .explode("item_id")
            .sort_values(["user_id", "sim"], ascending=False)
            .drop_duplicates(["user_id", "item_id"], keep="first")
            .merge(self.item_idf, left_on="item_id", right_on="index", how="left")
        )

        recs["score"] = recs["sim"] * recs["idf"]
        recs = recs.sort_values(["user_id", "score"], ascending=False)
        recs["rank"] = recs.groupby("user_id").cumcount() + 1
        return recs[recs["rank"] <= N_recs][["user_id", "item_id", "score", "rank"]]
    
    def recommend(self, dataset: Dataset, k: int = 10, users: list = [], **kwargs):
        """
        Provides N recommendations using KNN+popular (if needed)
        :param dataset: rectools.dataset.Dataset with users
        :param k: amount of recos
        :return: list of items ids
        """
        test = dataset.interactions.df

        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        mapper = self._generate_recs_mapper(
            model=self.user_knn,
            user_mapping=self.users_mapping,
            user_inv_mapping=self.users_inv_mapping,
            N=self.N_users,
        )

        # prepare similar users and their films for hot users
        hot_test = test[test["user_id"].isin(self.users_mapping)]
        recs = pd.DataFrame({"user_id": hot_test["user_id"].unique()})
        recs["sim_user_id"], recs["sim"] = zip(*recs["user_id"].map(mapper))
        recs = recs.set_index("user_id").apply(pd.Series.explode).reset_index()

        recs = recs[~(recs["user_id"] == recs["sim_user_id"])].merge(self.watched, on=["sim_user_id"], how="left")

        # prepare popular films for all users
        # since similarity is the low => we will take them only in case of need
        # 1) when the user is cold;
        # or 2) when there is lack of recs for hot user
        # cold_recs_list = [self.items_inv_mapping[reco] for reco in
        #                   self.recommend_popular(dataset,
        #                                          k=k)]
        cold_recs_list = self.recommend_popular(dataset, k=k)
        unique_users_list = np.unique(users)
        cold_recs = pd.DataFrame(
            {
                "user_id": unique_users_list,
                "sim_user_id": [-1 for _ in unique_users_list],
                "sim": [0.0001 for _ in unique_users_list],
                "item_id": [cold_recs_list for _ in unique_users_list],
            }
        )

        # concat all them, and proceed as before
        recs = pd.concat([recs, cold_recs])

        recs = (
            recs.explode("item_id")
            .sort_values(["user_id", "sim"], ascending=False)
            .drop_duplicates(["user_id", "item_id"], keep="first")
            .merge(self.item_idf, left_on="item_id", right_on="index", how="left")
        )

        recs["score"] = recs["sim"] * recs["idf"]
        recs = recs.sort_values(["user_id", "score"], ascending=False)
        recs["rank"] = recs.groupby("user_id").cumcount() + 1
        return recs[recs["rank"] <= k][["user_id", "item_id", "score", "rank"]]

    def recommend_popular(self, dataset, k: int = 10):
        """
        Provides N recommendations using popular model
        (for cold, and those, whose amount of recommendations <= k_recos)
        :param dataset:
        :param k: amount of recos
        :return: list of items ids
        """
        # extract popular list from popular_model
        sample_popular_user = dataset.user_id_map.get_external_sorted_by_internal()[0]
        recs = list(
            self.popular_model.recommend(dataset=dataset, users=[sample_popular_user], k=k, filter_viewed=False)[
                "item_id"
            ]
        )
        return recs

    def recommend_items_to_user(self, dataset: Dataset, user_id: int, k: int = 10):
        """
        For one user we need to use onl numpy to provide speed
        :param user_id: id of user
        :param k: amount of recos
        :return: list of recos
        """
        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        cold_recs = self.recommend_popular(dataset, k=k)

        # for cold users
        if user_id not in self.users_mapping_static:
            return cold_recs
        # for hot, predict, and append popular
        mapped_user_id = self.users_mapping_static[user_id]
        similar_users, sims = self.user_knn_static.similar_items(mapped_user_id, N=k)

        rec_items = []
        for sim_user, sim in zip(similar_users, sims):
            if sim_user == mapped_user_id:
                continue

            original_user_id = self.users_inv_mapping_static[sim_user]
            items_watched_by_sim_user = self.watched_static[original_user_id]
            for item in items_watched_by_sim_user:
                item_idf = self.item_idf_static[item]
                score = sim * item_idf
                rec_items.append((item, score))
        rec_items.sort(key=lambda x: x[1], reverse=False)

        recos = list(dict(rec_items).keys())[::-1]

        # be sure that unique
        cold_recs = [rec for rec in cold_recs if rec not in recos]
        if len(recos) < k:
            recos = recos + cold_recs[: k - len(recos)]
        else:
            recos = recos[:k]
        return recos