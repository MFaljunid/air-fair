"""
prepare_lastfm_final.py — the FINAL LastFM preprocessing recipe: apply the
agreed core filter (min_user=40, min_item=10 -- chosen for its close match
to FairIR's reported item count, 58,945 vs their 60,081), then subsample
down to 15,000 users (a compute-budget necessity -- FairIR's per-user
contrastive step is too slow to train in practical time at the full
~289,000-user filtered scale; timed at ~7 min/30 epochs at 15,000 users,
comparable to ML-1M's own training time).

This is a ONE-TIME check to confirm the final recipe produces the expected
sizes before running any real experiments on it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from preprocessing.load_data import load_and_prepare
from preprocessing.attributes import prepare_all_attributes

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "lastfm")

data = load_and_prepare(
    raw_dir=RAW_DIR, positive_threshold=None,
    min_user_interactions=40, min_item_interactions=10,
    subsample_users=15000, seed=0,
)
print(f"\nFINAL LastFM setup: n_users={data['n_users']}, n_items={data['n_items']}, "
      f"train={len(data['train'])}, test={len(data['test'])}")

attr_result = prepare_all_attributes(data["user_attrs"])
user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)
print("\nGender distribution:")
print(user_attrs_sorted["gender"].value_counts())
print("\nAge group distribution:")
print(user_attrs_sorted["age_group"].value_counts())