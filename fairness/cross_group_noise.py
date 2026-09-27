"""
cross_group_noise.py
---------------------
FairIR's "Inter-Group Knowledge Transfer" mechanism (Shi et al., KBS 2026,
Section 3.2, Equations 2-6). Reimplemented here as a standalone module,
backbone-agnostic (works with any embedding matrix, MF or GCCF).

Core idea: build each group's AGGREGATE positive-item pool (all items that
group has interacted with, across all its users), add learnable noise to
those items, then inject noised items from the OPPOSITE group into each
user's positive-sample set -- transferring preference "knowledge" bidirectionally
between groups to rebalance the data distribution in embedding space.

    Eq. (2)-(3): item noise
        v_tilde_i = v_i + eps_i,   eps_i ~ Uniform(-delta, delta)
        (initial noise is random; delta itself, and per-item eps once
        initialized, become LEARNABLE and are optimized jointly with the
        contrastive loss in contrastive_loss.py -- see note in __init__)

    Eq. (4): group embedding indexing
        v_u^m in R^d (male user embedding), v_u^f in R^d (female user embedding)

    Eq. (5): per-user real positive-item representations
        {<v_u^g, v_i^g> | u in group g, i in user u's own positive items}

    Eq. (6): bidirectional cross-group injection
        For a user u in group g, augment u's positive set with NOISED items
        drawn from the AGGREGATE positive-item pool of every OTHER group g'.
"""

import numpy as np
import pandas as pd


class CrossGroupNoise:
    def __init__(self, n_items: int, n_factors: int, delta: float = 0.1, seed: int = 0):
        """
        delta: initial noise magnitude, eps ~ Uniform(-delta, delta) -- Eq. (2)-(3).
        The noise array itself is a trainable parameter (updated via gradients
        in training/train_fairir.py, jointly with the contrastive loss), NOT
        resampled randomly on every call -- this matches the paper's "the
        direction and magnitude of the noise here are learnable and jointly
        optimized with the contrastive objective" (Section 3.2).
        """
        rng = np.random.default_rng(seed)
        self.noise = rng.uniform(-delta, delta, size=(n_items, n_factors))
        self.delta = delta

    def get_noised_item_emb(self, item_emb: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Eq. (2)-(3): v_tilde_i = v_i + eps_i, for the given item ids."""
        return item_emb[item_ids] + self.noise[item_ids]


def build_group_positive_sets(
    train_df: pd.DataFrame, user_attrs: pd.DataFrame, group_col: str = "gender_code"
) -> dict:
    """
    Aggregate positive-item pool per group: {group_value: np.array of unique
    item ids interacted with by ANY user in that group}. This is the
    group-level I^{+m}, I^{+f} used as the SOURCE pool for cross-group
    injection (Eq. 6), not any single user's personal positive set.
    """
    merged = train_df.merge(user_attrs[["user_id", group_col]], on="user_id", how="left")
    group_positive_sets = {}
    for group_value, sub in merged.groupby(group_col):
        group_positive_sets[group_value] = sub["item_id"].unique()
    return group_positive_sets


def sample_cross_group_items(
    own_group_value, group_positive_sets: dict, n_samples: int, rng
) -> np.ndarray:
    """
    For a user in own_group_value, sample n_samples item ids from the
    AGGREGATE positive pool of every OTHER group (bidirectional injection,
    Eq. 6). With exactly 2 groups (FairIR's binary gender case) this reduces
    to sampling purely from the single opposite group's pool; with more than
    2 groups (our later multi-group extension) it pools all other groups
    together as the injection source.
    """
    other_pools = [
        items for g, items in group_positive_sets.items() if g != own_group_value
    ]
    if not other_pools:
        return np.array([], dtype=np.int64)
    combined_pool = np.concatenate(other_pools)
    if len(combined_pool) == 0:
        return np.array([], dtype=np.int64)
    replace = len(combined_pool) < n_samples
    return rng.choice(combined_pool, size=n_samples, replace=replace)


if __name__ == "__main__":
    # Quick sanity check with synthetic data (no real files needed).
    rng = np.random.default_rng(0)
    n_users, n_items, n_factors = 20, 15, 8

    train_df = pd.DataFrame({
        "user_id": rng.integers(0, n_users, size=100),
        "item_id": rng.integers(0, n_items, size=100),
    }).drop_duplicates()

    user_attrs = pd.DataFrame({
        "user_id": np.arange(n_users),
        "gender_code": rng.integers(0, 2, size=n_users),  # 0=M, 1=F
    })

    group_positive_sets = build_group_positive_sets(train_df, user_attrs, "gender_code")
    print("Group positive-item pools:")
    for g, items in group_positive_sets.items():
        print(f"  group {g}: {len(items)} unique items -> {sorted(items)}")

    cgn = CrossGroupNoise(n_items, n_factors, delta=0.1, seed=0)
    item_emb = rng.normal(0, 1, size=(n_items, n_factors))

    # For a male user (group 0), sample cross-group (female-pool) items and noise them.
    sampled = sample_cross_group_items(0, group_positive_sets, n_samples=3, rng=rng)
    print(f"\nSampled cross-group items for a group-0 user: {sampled}")
    noised = cgn.get_noised_item_emb(item_emb, sampled)
    print(f"Original embeddings:\n{item_emb[sampled]}")
    print(f"Noised embeddings (should differ by at most delta={cgn.delta} per dim):\n{noised}")
    diff = np.abs(noised - item_emb[sampled])
    print(f"\nMax abs diff: {diff.max():.4f} (must be <= delta={cgn.delta})")
    assert diff.max() <= cgn.delta + 1e-9, "noise exceeded delta bound!"
    print("OK: noise correctly bounded by delta.")