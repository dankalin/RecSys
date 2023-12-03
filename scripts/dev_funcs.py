from pprint import pprint

import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm.auto import tqdm

from rectools import Columns
from rectools.dataset import Interactions, Dataset
from rectools.metrics import Precision, Recall, MeanInvUserFreq, Serendipity, MAP, MRR, MeanInvUserFreq,calc_metrics
from rectools.models import ImplicitItemKNNWrapperModel, RandomModel, PopularModel
from rectools.model_selection import TimeRangeSplitter
def calculate_metrics(models, metrics, splitter, K, interactions):
  results = []

  fold_iterator = splitter.split(interactions, collect_fold_stats=True)

  for train_ids, test_ids, fold_info in tqdm((fold_iterator), total=3):
      print(f"\n==================== Fold {fold_info['i_split']}")
      print(fold_info)

      df_train = interactions.df.iloc[train_ids]
      dataset = Dataset.construct(df_train)

      df_test = interactions.df.iloc[test_ids][Columns.UserItem]
      test_users = np.unique(df_test[Columns.User])

      catalog = df_train[Columns.Item].unique()

      for model_name, model in models.items():
          model = deepcopy(model)
          model.fit(dataset)
          recos = model.recommend(
              users=test_users,
              dataset=dataset,
              k=K,
              filter_viewed=True,
          )
          metric_values = calc_metrics(
              metrics,
              reco=recos,
              interactions=df_test,
              prev_interactions=df_train,
              catalog=catalog,
          )
          res = {"fold": fold_info["i_split"], "model": model_name}
          res.update(metric_values)
          results.append(res)
  return results

def get_visualize_recs(model, interactions, users, K, item_data):
  dataset = Dataset.construct(interactions)
  recommendations = model.recommend(users=users, dataset=dataset, k=K, filter_viewed=True)

  item_data_relevant = item_data[['item_id', 'content_type', 'title', 'title_orig', 'release_year', 'genres']]
  item_data_relevant['num_of_views'] = interactions.groupby('item_id')['user_id'].count()

  user_viewed_items_all = []
  user_recommendations_all = []

  for user_id in users:
    user_viewed_items = interactions[interactions['user_id'] == user_id].merge(item_data_relevant, on="item_id")
    user_recommendations = recommendations[recommendations['user_id'] == user_id].merge(item_data_relevant, on="item_id")

    user_viewed_items_all.append(user_viewed_items)
    user_recommendations_all.append(user_recommendations)

  viewed_items_dataset = pd.concat(user_viewed_items_all, ignore_index=True)
  recommendations_dataset = pd.concat(user_recommendations_all, ignore_index=True)

  return viewed_items_dataset, recommendations_dataset