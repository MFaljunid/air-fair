"""
check_lastfm_kcore.py — lightweight k-core threshold tuning: loads and
filters LastFM ONLY (no BPR training, no evaluation), so trying different
min_user_interactions/min_item_interactions values is fast to iterate on
while searching for a combination close to FairIR's reported LastFM stats
(139,371 users, 60,081 items, 4,017,311 interactions).

Edit MIN_USER_INTERACTIONS / MIN_ITEM_INTERACTIONS below and re-run.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from preprocessing.load_data import load_and_prepare

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "lastfm")

MIN_USER_INTERACTIONS = 45
MIN_ITEM_INTERACTIONS = 10

print(f"=== Loading LastFM (min_user={MIN_USER_INTERACTIONS}, min_item={MIN_ITEM_INTERACTIONS}) ===")
data = load_and_prepare(
    raw_dir=RAW_DIR, positive_threshold=None,
    min_user_interactions=MIN_USER_INTERACTIONS, min_item_interactions=MIN_ITEM_INTERACTIONS,
    seed=0,
)
print(f"\nFINAL: n_users={data['n_users']}, n_items={data['n_items']}, train={len(data['train'])}")
print("FairIR target: n_users=139371, n_items=60081, interactions=4017311")