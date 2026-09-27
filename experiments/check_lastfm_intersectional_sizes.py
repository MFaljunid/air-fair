"""
check_lastfm_intersectional_sizes.py — quick, cheap check (no training):
loads the final LastFM recipe (min_user=40, min_item=10, subsample=15000)
and reports the size of every intersectional (gender x age) group, so we
can pick a sensible target group BEFORE spending 30-40 min training on it.
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
attr_result = prepare_all_attributes(data["user_attrs"])
user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)

print(f"\nn_users={data['n_users']}")
print("\nIntersectional (gender x age) group sizes:")
counts = user_attrs_sorted.groupby(["group_label", "group_id"]).size().sort_values()
for (label, gid), n in counts.items():
    print(f"  {label:<10} (group_id={gid}): n={n}")

print(f"\nSmallest group: {counts.index[0][0]} with n={counts.iloc[0]} users")