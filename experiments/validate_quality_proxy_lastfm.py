"""
validate_quality_proxy_lastfm.py
------------------------------------
Same quality-proxy validation as validate_quality_proxy_v3.py (ML-1M), run
on LastFM's final recipe, so the two datasets can be compared side by side.
Uses the same 20%-per-user random holdout (multiple test items per user)
and compares uniform vs. popularity-weighted negative sampling for
quality(u) against true held-out multi-item test NDCG@10.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from preprocessing.load_data import load_and_prepare
from preprocessing.attributes import prepare_all_attributes
from models.mf import train_bpr
from fairness.instance_gate import compute_user_quality_proxy
from eval.metrics import get_top_k_per_user

SEED = 0
N_EPOCHS = 30
K = 10
TEST_SIZE = 0.2
MIN_TEST_ITEMS = 2
N_SAMPLES_PER_USER = 50

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
raw_dir = os.path.join(project_root, "data", "raw", "lastfm")
out_dir = os.path.join(project_root, "results", "lastfm", "diagnostics")
os.makedirs(out_dir, exist_ok=True)


def compute_user_quality_proxy_hard(model, train_df, n_users, n_items, n_samples_per_user=50, seed=0):
    rng = np.random.default_rng(seed)
    user_items = [[] for _ in range(n_users)]
    for u, i in zip(train_df["user_id"].values, train_df["item_id"].values):
        user_items[u].append(i)

    item_counts = train_df["item_id"].value_counts()
    pop = np.ones(n_items)
    pop[item_counts.index.values] += item_counts.values
    pop_prob = pop / pop.sum()

    quality = np.full(n_users, np.nan)
    for u in range(n_users):
        pos_items = user_items[u]
        if len(pos_items) == 0:
            continue
        n_samples = min(n_samples_per_user, len(pos_items))
        sampled_pos = rng.choice(pos_items, size=n_samples, replace=False)

        pos_set = set(pos_items)
        neg_items = []
        attempts = 0
        while len(neg_items) < n_samples and attempts < n_samples * 20:
            candidate = rng.choice(n_items, p=pop_prob)
            attempts += 1
            if candidate not in pos_set:
                neg_items.append(candidate)
        while len(neg_items) < n_samples:
            candidate = rng.integers(0, n_items)
            if candidate not in pos_set:
                neg_items.append(candidate)

        pos_scores = model.user_emb[u] @ model.item_emb[sampled_pos].T
        neg_scores = model.user_emb[u] @ model.item_emb[np.array(neg_items)].T
        quality[u] = np.mean(pos_scores > neg_scores)

    return quality


print("=== Loading LastFM (final recipe) with 20%-per-user random holdout ===")
data = load_and_prepare(
    raw_dir=raw_dir, positive_threshold=None,
    min_user_interactions=40, min_item_interactions=10,
    subsample_users=15000, split_method="random", test_size=TEST_SIZE, seed=SEED,
)
attr_result = prepare_all_attributes(data["user_attrs"])

print("\n=== Training plain BPR (30 epochs, seed=0) ===")
model = train_bpr(data["train"], data["n_users"], data["n_items"],
                   n_factors=64, n_epochs=N_EPOCHS, seed=SEED, verbose=False)

print("\n=== Computing quality(u): UNIFORM vs POPULARITY-WEIGHTED negatives ===")
quality_uniform = compute_user_quality_proxy(
    model, data["train"], data["n_users"], data["n_items"],
    n_samples_per_user=N_SAMPLES_PER_USER, seed=SEED,
)
quality_hard = compute_user_quality_proxy_hard(
    model, data["train"], data["n_users"], data["n_items"],
    n_samples_per_user=N_SAMPLES_PER_USER, seed=SEED,
)
print(f"Uniform-negative quality:      mean={np.nanmean(quality_uniform):.4f}  std={np.nanstd(quality_uniform):.4f}")
print(f"Popularity-weighted quality:   mean={np.nanmean(quality_hard):.4f}  std={np.nanstd(quality_hard):.4f}")

print(f"\n=== Computing per-user test-time NDCG@{K} over MULTIPLE held-out items/user ===")
top_k = get_top_k_per_user(model, data["train"], data["n_users"], data["n_items"], k=K)
user_test_items = [[] for _ in range(data["n_users"])]
for u, i in zip(data["test"]["user_id"].values, data["test"]["item_id"].values):
    user_test_items[u].append(i)

test_ndcg = np.full(data["n_users"], np.nan)
n_test_items_arr = np.zeros(data["n_users"], dtype=int)
for u in range(data["n_users"]):
    held_out = set(user_test_items[u])
    n_test_items_arr[u] = len(held_out)
    if len(held_out) < MIN_TEST_ITEMS:
        continue
    ranked = top_k[u]
    dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(ranked) if item in held_out)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(K, len(held_out))))
    test_ndcg[u] = dcg / idcg if idcg > 0 else 0.0

valid = (~np.isnan(quality_uniform)) & (~np.isnan(quality_hard)) & (~np.isnan(test_ndcg))
ndcg_valid = test_ndcg[valid]
print(f"\nValid users: {valid.sum()} / {data['n_users']}")
print(f"Mean held-out items per valid user: {n_test_items_arr[valid].mean():.1f}")

for label, q in [("UNIFORM negatives", quality_uniform), ("POPULARITY-WEIGHTED negatives", quality_hard)]:
    q_valid = q[valid]
    rho, p_s = spearmanr(q_valid, ndcg_valid)
    r, p_p = pearsonr(q_valid, ndcg_valid)
    print(f"\n=== {label} ===")
    print(f"Spearman rho = {rho:.4f}  (p = {p_s:.2e})")
    print(f"Pearson  r   = {r:.4f}  (p = {p_p:.2e})")
    if np.isnan(q_valid).all() or np.nanstd(q_valid) == 0:
        print("  (quality(u) is constant across users on this run -- skipping decile breakdown)")
        continue
    deciles = pd.qcut(q_valid, 10, labels=False, duplicates="drop")
    for d in sorted(np.unique(deciles[~np.isnan(deciles)]).astype(int)):
        mask = deciles == d
        print(f"  Decile {int(d)+1:2d}: mean_quality={q_valid[mask].mean():.4f}  "
              f"mean_test_NDCG@{K}={ndcg_valid[mask].mean():.4f}  (n={int(mask.sum())})")

pd.DataFrame({
    "user_id": np.where(valid)[0],
    "quality_uniform": quality_uniform[valid],
    "quality_hard": quality_hard[valid],
    "test_ndcg": ndcg_valid,
}).to_csv(os.path.join(out_dir, "quality_proxy_validation_lastfm_raw.csv"), index=False)

print(f"\nSaved: {out_dir}/quality_proxy_validation_lastfm_raw.csv")
print("\n(Run make_quality_proxy_dual_figure.py afterward to build the combined ML-1M + LastFM figure.)")