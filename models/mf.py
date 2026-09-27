"""
mf.py
-----
Basic Matrix Factorization trained with BPR (Bayesian Personalized Ranking)
loss. This is the backbone model for CD-AFMS: after training, the learned
user embeddings serve as the latent-preference proxy U fed into the causal
diagnosis pipeline (causal_diagnosis/multi_group_diagnosis.py), replacing
the random placeholder U used so far.

Pure NumPy implementation (no PyTorch dependency) -- simple enough that a
vectorized mini-batch SGD loop is fast and dependency-light for the pilot.

BPR objective (Rendle et al. 2009):
    For each observed (user u, positive item i) pair, sample a random
    unobserved item j for that user, and push the model to rank i above j:
        L = -sum log(sigmoid(x_ui - x_uj)) + reg * (||p_u||^2 + ||q_i||^2 + ||q_j||^2)
    where x_ui = p_u . q_i  (dot product of user/item embeddings).
"""

import numpy as np
import pandas as pd


class MatrixFactorization:
    def __init__(self, n_users: int, n_items: int, n_factors: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 0.01
        self.n_factors = n_factors
        self.user_emb = rng.normal(0, scale, size=(n_users, n_factors))
        self.item_emb = rng.normal(0, scale, size=(n_items, n_factors))

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Dot-product score for aligned arrays of user_ids/item_ids."""
        return np.sum(self.user_emb[user_ids] * self.item_emb[item_ids], axis=1)

    def predict_all_items(self, user_id: int) -> np.ndarray:
        """Score a single user against every item (for ranking-based eval later)."""
        return self.item_emb @ self.user_emb[user_id]


def _build_user_positive_items(train_df: pd.DataFrame, n_users: int) -> list:
    """List (indexed by user_id) of numpy arrays of that user's positive item ids."""
    user_items = [[] for _ in range(n_users)]
    for u, i in zip(train_df["user_id"].values, train_df["item_id"].values):
        user_items[u].append(i)
    return [np.array(items, dtype=np.int64) for items in user_items]


def _sample_negatives(
    user_ids: np.ndarray, user_positive_items: list, n_items: int, rng
) -> np.ndarray:
    """For each user in user_ids, sample one item they have NOT interacted with."""
    neg_items = np.empty(len(user_ids), dtype=np.int64)
    for idx, u in enumerate(user_ids):
        pos_set = user_positive_items[u]
        while True:
            candidate = rng.integers(0, n_items)
            if candidate not in pos_set:
                neg_items[idx] = candidate
                break
    return neg_items


def train_bpr(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    n_factors: int = 64,
    n_epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 0.05,
    reg: float = 0.01,
    seed: int = 0,
    verbose: bool = True,
) -> MatrixFactorization:
    """
    Train a MatrixFactorization model with mini-batch BPR-SGD.
    Returns the trained model (model.user_emb is the U proxy for causal diagnosis).
    """
    rng = np.random.default_rng(seed)
    model = MatrixFactorization(n_users, n_items, n_factors=n_factors, seed=seed)
    user_positive_items = _build_user_positive_items(train_df, n_users)

    interactions_u = train_df["user_id"].values
    interactions_i = train_df["item_id"].values
    n_interactions = len(interactions_u)

    for epoch in range(n_epochs):
        perm = rng.permutation(n_interactions)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_interactions, batch_size):
            batch_idx = perm[start : start + batch_size]
            u = interactions_u[batch_idx]
            i_pos = interactions_i[batch_idx]
            i_neg = _sample_negatives(u, user_positive_items, n_items, rng)

            p_u = model.user_emb[u]              # (B, F)
            q_pos = model.item_emb[i_pos]         # (B, F)
            q_neg = model.item_emb[i_neg]         # (B, F)

            x_uij = np.sum(p_u * (q_pos - q_neg), axis=1)   # (B,)
            sigmoid_neg = 1.0 / (1.0 + np.exp(x_uij))       # sigmoid(-x_uij), stable form
            loss = -np.log(1.0 / (1.0 + np.exp(-x_uij)) + 1e-12)
            epoch_loss += loss.sum()
            n_batches += 1

            # Gradients of BPR loss w.r.t. embeddings (scaled by sigmoid(-x_uij))
            grad_common = sigmoid_neg[:, None]  # (B, 1)
            grad_p_u = -grad_common * (q_pos - q_neg) + reg * p_u
            grad_q_pos = -grad_common * p_u + reg * q_pos
            grad_q_neg = grad_common * p_u + reg * q_neg

            # Scatter-add updates since the same user/item can repeat within a batch.
            np.add.at(model.user_emb, u, -lr * grad_p_u)
            np.add.at(model.item_emb, i_pos, -lr * grad_q_pos)
            np.add.at(model.item_emb, i_neg, -lr * grad_q_neg)

        if verbose:
            print(f"epoch {epoch + 1}/{n_epochs}  avg_bpr_loss={epoch_loss / n_interactions:.4f}")

    return model


if __name__ == "__main__":
    # Quick check: trains on the real ML-100K data (requires load_data.py's
    # raw files to already be in place) and reports the BPR loss trend plus
    # a sanity check that user embeddings differ meaningfully across users.
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from preprocessing.load_data import load_and_prepare

    data = load_and_prepare()
    print(f"Training MF on {data['n_users']} users, {data['n_items']} items, "
          f"{len(data['train'])} train interactions...")

    model = train_bpr(
        data["train"], data["n_users"], data["n_items"],
        n_factors=64, n_epochs=15, batch_size=1024, lr=0.05, reg=0.01, seed=0,
    )

    print()
    print("user_emb shape:", model.user_emb.shape)
    print("mean embedding norm:", np.linalg.norm(model.user_emb, axis=1).mean())
    print("std across users (per-dim avg):", model.user_emb.std(axis=0).mean())
