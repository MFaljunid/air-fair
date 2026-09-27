"""
metrics.py
----------
Full-ranking evaluation for Leave-One-Out (LOO) protocol: for each user with
one held-out test interaction, rank ALL items (excluding items already seen
in train) and compute HR@K / NDCG@K based on where the true test item lands
in that ranking.

This deliberately avoids the "sampled negatives" (e.g. 1 positive + 99
random negatives) protocol -- see Krichene & Rendle (KDD 2020), "On Sampled
Metrics for Item Recommendation", which shows sampled metrics can even
reverse the true ranking of models.

Memory note: for large catalogs (e.g. LastFM's ~60K items x ~139K users),
computing the full (n_users x n_items) score matrix at once would need tens
of GB. evaluate_full_ranking() therefore scores users in BATCHES, not all
at once.
"""

import numpy as np
import pandas as pd


def _build_user_train_items(train_df: pd.DataFrame, n_users: int) -> list:
    """List of sets: item ids each user has already interacted with in train."""
    user_items = [set() for _ in range(n_users)]
    for u, i in zip(train_df["user_id"].values, train_df["item_id"].values):
        user_items[u].add(i)
    return user_items


def _build_user_test_item(test_df: pd.DataFrame, n_users: int) -> np.ndarray:
    """
    Array of length n_users: the held-out test item id for each user, or -1
    if that user has no test interaction (shouldn't happen after LOO, but
    guarded against).
    """
    test_item = np.full(n_users, -1, dtype=np.int64)
    for u, i in zip(test_df["user_id"].values, test_df["item_id"].values):
        test_item[u] = i  # LOO guarantees exactly one test row per user
    return test_item


def evaluate_full_ranking(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    k_list: list = (10, 20, 30, 40),
    batch_size: int = 512,
) -> dict:
    """
    Full-ranking HR@K and NDCG@K under the LOO protocol.

    model must expose:
        model.user_emb : (n_users, d) array
        model.item_emb : (n_items, d) array
    (matches models/mf.py's MatrixFactorization; a GCCF backbone should
    expose the same two attributes after its own forward pass).

    Returns a dict: {"HR@10": ..., "NDCG@10": ..., "HR@20": ..., ...}
    """
    user_train_items = _build_user_train_items(train_df, n_users)
    test_item = _build_user_test_item(test_df, n_users)

    max_k = max(k_list)
    hits = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}

    valid_users = np.where(test_item >= 0)[0]

    for start in range(0, len(valid_users), batch_size):
        batch_users = valid_users[start : start + batch_size]

        # (batch, d) @ (d, n_items) -> (batch, n_items) score matrix.
        scores = model.user_emb[batch_users] @ model.item_emb.T

        for row_idx, u in enumerate(batch_users):
            # Mask out items already seen in train so they can't be "recommended".
            seen = user_train_items[u]
            if seen:
                scores[row_idx, list(seen)] = -np.inf

        # Top-max_k item ids per row, sorted by descending score.
        top_k_idx = np.argpartition(-scores, kth=min(max_k, scores.shape[1] - 1), axis=1)[:, :max_k]
        for row_idx, u in enumerate(batch_users):
            row_top = top_k_idx[row_idx]
            row_scores = scores[row_idx, row_top]
            order = np.argsort(-row_scores)
            ranked_items = row_top[order]

            true_item = test_item[u]
            rank_positions = np.where(ranked_items == true_item)[0]
            rank = rank_positions[0] if len(rank_positions) > 0 else None

            for k in k_list:
                hit = 1 if (rank is not None and rank < k) else 0
                hits[k].append(hit)
                ndcg = (1.0 / np.log2(rank + 2)) if (rank is not None and rank < k) else 0.0
                ndcgs[k].append(ndcg)

    results = {}
    for k in k_list:
        results[f"HR@{k}"] = float(np.mean(hits[k]))
        results[f"NDCG@{k}"] = float(np.mean(ndcgs[k]))
    return results


def get_top_k_per_user(
    model, train_df: pd.DataFrame, n_users: int, n_items: int,
    k: int, batch_size: int = 512,
) -> dict:
    """
    Returns {user_id: np.array of top-k item ids (excluding train items)},
    for every user (not just those with a test interaction) -- needed for
    group-level fairness metrics (DP, EO) which aggregate over all users in
    each sensitive-attribute group.
    """
    user_train_items = _build_user_train_items(train_df, n_users)
    top_k_by_user = {}

    for start in range(0, n_users, batch_size):
        batch_users = np.arange(start, min(start + batch_size, n_users))
        scores = model.user_emb[batch_users] @ model.item_emb.T

        for row_idx, u in enumerate(batch_users):
            seen = user_train_items[u]
            if seen:
                scores[row_idx, list(seen)] = -np.inf

        top_k_idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
        for row_idx, u in enumerate(batch_users):
            row_top = top_k_idx[row_idx]
            row_scores = scores[row_idx, row_top]
            order = np.argsort(-row_scores)
            top_k_by_user[int(u)] = row_top[order]

    return top_k_by_user


def demographic_parity(
    top_k_by_user: dict, user_group: np.ndarray, n_items: int,
) -> float:
    """
    FairIR Eq. (19): DP = (1/|V|) * sum_v | count_G0(v) - count_G1(v) | / (count_G0(v) + count_G1(v))
    where count_Gg(v) = number of users in group g whose top-K list contains item v.
    Only defined for a BINARY user_group array (0/1) -- matches FairIR's own
    binary-gender evaluation exactly, for a direct baseline comparison.
    Lower is better (0 = perfectly equal exposure across groups).
    """
    count_g0 = np.zeros(n_items)
    count_g1 = np.zeros(n_items)

    for u, top_k in top_k_by_user.items():
        if user_group[u] == 0:
            count_g0[top_k] += 1
        else:
            count_g1[top_k] += 1

    denom = count_g0 + count_g1
    mask = denom > 0  # items recommended to nobody don't contribute (0/0 undefined)
    dp = np.abs(count_g0[mask] - count_g1[mask]) / denom[mask]
    return float(dp.sum() / n_items)  # normalized by |V| = n_items, matching Eq. (19)


def equal_opportunity(
    top_k_by_user: dict, user_group: np.ndarray, test_item_by_user: np.ndarray, n_items: int,
) -> float:
    """
    FairIR Eq. (20): same as DP but restricted to (item, user) pairs where the
    item is BOTH in the user's top-K AND is that user's true held-out test
    item (i.e. counts only genuine hits, not all recommended items).
    Lower is better.
    """
    count_g0 = np.zeros(n_items)
    count_g1 = np.zeros(n_items)

    for u, top_k in top_k_by_user.items():
        true_item = test_item_by_user[u]
        if true_item < 0:
            continue  # no test interaction for this user
        if true_item in top_k:
            if user_group[u] == 0:
                count_g0[true_item] += 1
            else:
                count_g1[true_item] += 1

    denom = count_g0 + count_g1
    mask = denom > 0
    eo = np.abs(count_g0[mask] - count_g1[mask]) / denom[mask]
    return float(eo.sum() / n_items)


def evaluate_fairness(
    model, train_df: pd.DataFrame, test_df: pd.DataFrame,
    user_group: np.ndarray, n_users: int, n_items: int,
    k_list: list = (10, 20, 30, 40), batch_size: int = 512,
) -> dict:
    """
    Convenience wrapper: computes DP@K and EO@K for each K in k_list.
    user_group must be a length-n_users array of 0/1 (e.g. gender_code from
    preprocessing/attributes.py).
    """
    test_item_by_user = _build_user_test_item(test_df, n_users)
    results = {}
    for k in k_list:
        top_k_by_user = get_top_k_per_user(model, train_df, n_users, n_items, k, batch_size)
        results[f"DP@{k}"] = demographic_parity(top_k_by_user, user_group, n_items)
        results[f"EO@{k}"] = equal_opportunity(top_k_by_user, user_group, test_item_by_user, n_items)
    return results


if __name__ == "__main__":
    # Quick check on the real ML-1M data (requires data/raw/ml-1m/ to be populated).
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from preprocessing.load_data import load_and_prepare
    from preprocessing.attributes import prepare_all_attributes
    from models.mf import train_bpr

    data = load_and_prepare(raw_dir=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "ml-1m"
    ))
    print(f"n_users={data['n_users']}, n_items={data['n_items']}")

    model = train_bpr(
        data["train"], data["n_users"], data["n_items"],
        n_factors=64, n_epochs=15, seed=0,
    )

    results = evaluate_full_ranking(
        model, data["train"], data["test"], data["n_users"], data["n_items"],
        k_list=[10, 20, 30, 40],
    )
    print("\n=== Full-ranking evaluation (LOO) ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print("\n=== Fairness evaluation (gender, binary) ===")
    attr_result = prepare_all_attributes(data["user_attrs"])
    user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)
    user_group = user_attrs_sorted["gender_code"].values  # 0=M, 1=F

    fairness_results = evaluate_fairness(
        model, data["train"], data["test"], user_group,
        data["n_users"], data["n_items"], k_list=[10, 20, 30, 40],
    )
    for metric, value in fairness_results.items():
        print(f"{metric}: {value:.4f}")