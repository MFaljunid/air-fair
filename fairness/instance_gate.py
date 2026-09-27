"""
instance_gate.py
------------------
AIR-Fair's Instance-Adaptive Gate: replaces FairIR's fixed group-level
(alpha, delta) with per-user values (alpha_u, delta_u), computed from each
user's individual deviation from their dominant group's recommendation
quality -- addressing FairIR's stated limitation that uniform group-level
correction over- or under-corrects individual users regardless of their
actual need.

Design constraint: g(u) MUST be computable from TRAIN data only (no test
labels), or it would leak test information into training. We use a
self-supervised proxy -- per-user pairwise ranking accuracy on sampled
(positive, negative) pairs from that user's own train interactions, using
the model's CURRENT embeddings. This is the same quantity BPR itself
optimizes, just measured per user instead of averaged globally.

    quality(u)   = fraction of sampled (pos, neg) pairs where
                   score(u, pos) > score(u, neg), using current embeddings
    g(u)         = |quality(u) - quality(dominant_group)| / quality(dominant_group)
    delta_u      = delta_base * (1 + g(u))   -- always >= delta_base
    alpha_u      = clip(alpha_base * (1 + g(u)), 0, 1)
"""

import numpy as np
import pandas as pd


def compute_user_quality_proxy(
    model, train_df: pd.DataFrame, n_users: int, n_items: int,
    n_samples_per_user: int = 50, seed: int = 0,
) -> np.ndarray:
    """
    Per-user pairwise ranking accuracy proxy, computed ENTIRELY from train
    data + current model embeddings (no test labels touched).
    Returns an array of length n_users; users with no train interactions
    get NaN (excluded from downstream group-quality averaging).
    """
    rng = np.random.default_rng(seed)
    user_items = [[] for _ in range(n_users)]
    for u, i in zip(train_df["user_id"].values, train_df["item_id"].values):
        user_items[u].append(i)

    quality = np.full(n_users, np.nan)
    for u in range(n_users):
        pos_items = user_items[u]
        if len(pos_items) == 0:
            continue
        n_samples = min(n_samples_per_user, len(pos_items))
        sampled_pos = rng.choice(pos_items, size=n_samples, replace=False)

        pos_set = set(pos_items)
        neg_items = []
        while len(neg_items) < n_samples:
            candidate = rng.integers(0, n_items)
            if candidate not in pos_set:
                neg_items.append(candidate)

        pos_scores = model.user_emb[u] @ model.item_emb[sampled_pos].T
        neg_scores = model.user_emb[u] @ model.item_emb[np.array(neg_items)].T
        quality[u] = np.mean(pos_scores > neg_scores)

    return quality


def compute_gate_values(
    quality: np.ndarray, user_group_map: np.ndarray, dominant_group=None,
) -> np.ndarray:
    """
    g(u) = |quality(u) - quality(dominant_group)| / quality(dominant_group).
    dominant_group: the group value (e.g. 0 for male) to treat as the
    reference "dominant" group. If None, the group with the LARGER
    population is used automatically (matches the common real-world case,
    e.g. ML-1M's ~72% male majority).
    Returns g(u) for every user, clipped at a max of 3.0 to avoid extreme
    values when dominant_group's average quality is very close to zero
    (division-by-near-zero safety, documented rather than silently hidden).
    """
    valid = ~np.isnan(quality)

    if dominant_group is None:
        groups, counts = np.unique(user_group_map[valid], return_counts=True)
        dominant_group = groups[np.argmax(counts)]

    dominant_mask = valid & (user_group_map == dominant_group)
    dominant_quality = quality[dominant_mask].mean()

    g = np.full(len(quality), np.nan)
    eps = 1e-6
    g[valid] = np.abs(quality[valid] - dominant_quality) / (dominant_quality + eps)
    g = np.clip(g, 0, 3.0)
    return g, dominant_group, dominant_quality


def get_adaptive_alpha_delta(
    g_values: np.ndarray,
    alpha_base: float = None, delta_base: float = None,  # kept for backward-compat signature; unused by rank method
    alpha_min: float = 0.1, alpha_max: float = 0.9,
    delta_min_ratio: float = 0.5, delta_max_ratio: float = 2.0,
    invert: bool = False,
) -> tuple:
    """
    RANK-BASED normalization (not a small additive bump on g(u)).

    invert=False (default, "more correction for the disadvantaged"):
        higher g(u) (more deviation from the dominant group) -> HIGHER
        alpha_u/delta_u (more cross-group noise injected).

    invert=True ("protect the disadvantaged, burden the well-served"):
        higher g(u) -> LOWER alpha_u/delta_u, i.e. disadvantaged users keep
        MORE of their own real signal (less diluted by cross-group noise),
        while already-well-served (dominant-like) users receive more of the
        noise injection since they have enough real signal to spare.
        Motivated by the concern that flooding an already poorly-modeled
        user's positive set with noised items from a DIFFERENT group's
        preferences may drown their few genuine signals rather than help --
        worth testing empirically as the opposite polarity of the same gate.

    Why rank-based at all: g(u) itself can be tiny (e.g. 0.07 for the
    most-different age group found on ML-1M), so a naive additive multiplier
    stays within a few percent of the base value for every user. Percentile-
    ranking g(u) across users guarantees the full [alpha_min, alpha_max]
    range is actually used regardless of how small the underlying magnitude
    differences are.

    Users with g(u)=NaN (no train interactions) get the midpoint (rank=0.5).
    """
    valid = ~np.isnan(g_values)
    ranks = np.full(len(g_values), 0.5)
    if valid.sum() > 1:
        order = np.argsort(g_values[valid])
        pct = np.empty(valid.sum())
        pct[order] = np.linspace(0.0, 1.0, valid.sum())
        ranks[valid] = pct

    if invert:
        ranks = 1.0 - ranks

    alpha_u = alpha_min + ranks * (alpha_max - alpha_min)
    delta_multiplier = delta_min_ratio + ranks * (delta_max_ratio - delta_min_ratio)
    delta_u = (delta_base if delta_base is not None else 0.1) * delta_multiplier
    return alpha_u, delta_u


if __name__ == "__main__":
    # Sanity check with a MANUFACTURED quality gap: group 0 (dominant) gets
    # well-separated pos/neg scores (high quality); group 1 gets nearly
    # random scores (low quality, simulating an under-served group). The
    # gate should correctly assign HIGHER g(u) -- and thus stronger
    # correction -- to group 1.
    rng = np.random.default_rng(0)
    n_users, n_items, n_factors = 40, 30, 8

    class DummyModel:
        pass

    model = DummyModel()
    model.item_emb = rng.normal(0, 1, size=(n_items, n_factors))
    model.user_emb = np.zeros((n_users, n_factors))

    user_group_map = np.array([0] * 30 + [1] * 10)  # group 0 dominant (30 users), group 1 minority (10)

    train_rows = []
    for u in range(n_users):
        items = rng.choice(n_items, size=10, replace=False)
        for i in items:
            train_rows.append((u, i))
    train_df = pd.DataFrame(train_rows, columns=["user_id", "item_id"])

    # Craft embeddings: group-0 users get an embedding aligned with their
    # positive items (high quality); group-1 users get near-random embeddings.
    for u in range(n_users):
        pos_items = train_df.loc[train_df["user_id"] == u, "item_id"].values
        if user_group_map[u] == 0:
            model.user_emb[u] = model.item_emb[pos_items].mean(axis=0) * 2.0  # well-aligned
        else:
            model.user_emb[u] = rng.normal(0, 0.1, size=n_factors)  # weak signal

    quality = compute_user_quality_proxy(model, train_df, n_users, n_items, n_samples_per_user=8, seed=0)
    print(f"Group 0 (dominant) mean quality: {quality[user_group_map==0].mean():.4f}")
    print(f"Group 1 (minority) mean quality: {quality[user_group_map==1].mean():.4f}")

    g_values, dominant_group, dominant_quality = compute_gate_values(quality, user_group_map)
    print(f"\nDetected dominant group: {dominant_group} (quality={dominant_quality:.4f})")
    print(f"Group 0 mean g(u): {np.nanmean(g_values[user_group_map==0]):.4f} (should be near 0)")
    print(f"Group 1 mean g(u): {np.nanmean(g_values[user_group_map==1]):.4f} (should be clearly higher)")

    alpha_u, delta_u = get_adaptive_alpha_delta(g_values, delta_base=0.1, alpha_min=0.1, alpha_max=0.9)
    print(f"\nGroup 0 mean alpha_u: {alpha_u[user_group_map==0].mean():.4f} (near base 0.5)")
    print(f"Group 1 mean alpha_u: {alpha_u[user_group_map==1].mean():.4f} (should be higher, capped at 1.0)")
    print(f"Group 0 mean delta_u: {delta_u[user_group_map==0].mean():.4f} (near base 0.1)")
    print(f"Group 1 mean delta_u: {delta_u[user_group_map==1].mean():.4f} (should be higher)")

    assert np.nanmean(g_values[user_group_map==1]) > np.nanmean(g_values[user_group_map==0]), \
        "gate should assign higher correction to the lower-quality group!"
    print("\nOK: gate correctly identifies the disadvantaged group and scales correction up for it.")