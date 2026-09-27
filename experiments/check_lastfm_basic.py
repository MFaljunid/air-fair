"""
check_lastfm_basic.py — first sanity check on LastFM, mirroring the very
first check we did on ML-1M: load the data, confirm sizes make sense, train
a QUICK plain BPR (5 epochs, not 30) just to confirm the pipeline runs on
this dataset's different shape (no timestamps -> random split fallback,
implicit play-count 'rating' -> positive_threshold=None, no 1-5 filter).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from preprocessing.load_data import load_and_prepare
from preprocessing.attributes import prepare_all_attributes
from models.mf import train_bpr
from eval.metrics import evaluate_full_ranking, evaluate_fairness

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "lastfm")

print("=== Loading LastFM (no positive_threshold -- play counts aren't a 1-5 scale) ===")
data = load_and_prepare(raw_dir=RAW_DIR, positive_threshold=None, seed=0)
print(f"n_users={data['n_users']}, n_items={data['n_items']}, "
      f"train={len(data['train'])}, test={len(data['test'])}")

attr_result = prepare_all_attributes(data["user_attrs"])
user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)
print("\nGender distribution:")
print(user_attrs_sorted["gender"].value_counts())
print("\nAge group distribution:")
print(user_attrs_sorted["age_group"].value_counts())

print("\n=== Quick BPR sanity (5 epochs only) ===")
model = train_bpr(
    data["train"], data["n_users"], data["n_items"],
    n_factors=64, n_epochs=5, seed=0,
)

print("\n=== Quick eval (K=10 only) ===")
acc = evaluate_full_ranking(model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=[10])
user_group = user_attrs_sorted["gender_code"].values
fair = evaluate_fairness(model, data["train"], data["test"], user_group, data["n_users"], data["n_items"], k_list=[10])
print(acc, fair)
print("\nDone -- pipeline runs on LastFM. any NaN:", np.isnan(model.user_emb).any())