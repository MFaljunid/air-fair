"""
validate_quality_proxy_v3.py
--------------------------------
Option B fix for the quality-proxy validation gap found in v1/v2: the
original quality(u) samples UNIFORM random negatives, which makes the
pairwise task too easy (near-saturated at ~0.9-1.0 for almost everyone),
while the real full-ranking test task competes against the ENTIRE catalog,
dominated in practice by popular items. This mismatch in task difficulty
is a plausible reason quality(u) failed to correlate with real held-out
performance (v1: rho=+0.042, v2: rho=-0.055).

Fix: recompute quality(u) using POPULARITY-WEIGHTED negative sampling
(negatives drawn proportional to how often each item appears in train)
instead of uniform random negatives. Popular items are harder to beat in
a pairwise comparison (closer to what a user's true positive item must
outrank in full ranking), which should better reflect true ranking
difficulty and, if the hypothesis is right, correlate more strongly with
real test-time NDCG.

Reuses the exact same 20%-per-user split, trained model, and multi-item
test NDCG ground truth as v2, changing ONLY how quality(u) itself is
computed, so the two rho values are directly comparable.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
raw_dir = os.path.join(project_root, "data", "raw", "ml-1m")
out_dir = os.path.join(project_root, "results", "ml-1m", "diagnostics")
os.makedirs(out_dir, exist_ok=True)


def compute_user_quality_proxy_hard(model, train_df, n_users, n_items, n_samples_per_user=50, seed=0):
    """Same as compute_user_quality_proxy, but negatives are sampled
    proportional to item popularity in train_df (harder negatives) instead
    of uniformly at random."""
    rng = np.random.default_rng(seed)
    user_items = [[] for _ in range(n_users)]
    for u, i in zip(train_df["user_id"].values, train_df["item_id"].values):
        user_items[u].append(i)

    item_counts = train_df["item_id"].value_counts()
    pop = np.ones(n_items)  # smoothing floor so every item has nonzero prob
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
        while len(neg_items) < n_samples:  # fallback if popular items are exhausted by pos_set
            candidate = rng.integers(0, n_items)
            if candidate not in pos_set:
                neg_items.append(candidate)

        pos_scores = model.user_emb[u] @ model.item_emb[sampled_pos].T
        neg_scores = model.user_emb[u] @ model.item_emb[np.array(neg_items)].T
        quality[u] = np.mean(pos_scores > neg_scores)

    return quality


print("=== Loading ML-1M with 20%-per-user random holdout (same as v2) ===")
data = load_and_prepare(raw_dir=raw_dir, positive_threshold=3, split_method="random",
                         test_size=TEST_SIZE, seed=SEED)
attr_result = prepare_all_attributes(data["user_attrs"])

print("\n=== Training plain BPR (30 epochs, seed=0) ===")
model = train_bpr(data["train"], data["n_users"], data["n_items"],
                   n_factors=64, n_epochs=N_EPOCHS, seed=SEED, verbose=False)

print("\n=== Computing quality(u): UNIFORM negatives (original) vs POPULARITY-WEIGHTED negatives (v3 fix) ===")
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

print(f"\n=== Computing per-user test-time NDCG@{K} over MULTIPLE held-out items/user (same as v2) ===")
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

for label, q in [("UNIFORM negatives (original)", quality_uniform), ("POPULARITY-WEIGHTED negatives (v3 fix)", quality_hard)]:
    q_valid = q[valid]
    rho, p_s = spearmanr(q_valid, ndcg_valid)
    r, p_p = pearsonr(q_valid, ndcg_valid)
    print(f"\n=== {label} ===")
    print(f"Spearman rho = {rho:.4f}  (p = {p_s:.2e})")
    print(f"Pearson  r   = {r:.4f}  (p = {p_p:.2e})")

    deciles = pd.qcut(q_valid, 10, labels=False, duplicates="drop")
    for d in sorted(np.unique(deciles)):
        mask = deciles == d
        print(f"  Decile {int(d)+1:2d}: mean_quality={q_valid[mask].mean():.4f}  "
              f"mean_test_NDCG@{K}={ndcg_valid[mask].mean():.4f}  (n={int(mask.sum())})")

pd.DataFrame({
    "user_id": np.where(valid)[0],
    "quality_uniform": quality_uniform[valid],
    "quality_hard": quality_hard[valid],
    "test_ndcg": ndcg_valid,
}).to_csv(os.path.join(out_dir, "quality_proxy_validation_v3_raw.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, q, label in [(axes[0], quality_uniform[valid], "Uniform negatives (original)"),
                       (axes[1], quality_hard[valid], "Popularity-weighted negatives (v3 fix)")]:
    rho, p_s = spearmanr(q, ndcg_valid)
    ax.scatter(q, ndcg_valid, s=4, alpha=0.15, color="tab:blue")
    ax.set_xlabel("quality(u)")
    ax.set_ylabel(f"Test-time NDCG@{K}")
    ax.set_title(f"{label}\nSpearman rho={rho:.3f}, p={p_s:.1e}")
    ax.grid(alpha=0.2)
fig.suptitle("Quality Proxy: Uniform vs. Popularity-Weighted Negatives (ML-1M, seed=0)", fontsize=11)
fig.tight_layout(pad=1.5)
fig.savefig(os.path.join(out_dir, "quality_proxy_validation_v3.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "quality_proxy_validation_v3.pdf"), bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {out_dir}/quality_proxy_validation_v3_raw.csv")
print(f"Saved: {out_dir}/quality_proxy_validation_v3.png (+.pdf)")