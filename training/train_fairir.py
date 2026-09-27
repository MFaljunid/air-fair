"""
train_fairir.py
------------------
Full FairIR training loop (Shi et al., KBS 2026), reimplemented as our
baseline. Combines:
    - models/mf.py               : backbone embeddings (theta), trained via BPR
    - fairness/cross_group_noise.py : learnable noise (delta) + cross-group injection
    - fairness/contrastive_loss.py  : L_info (fairness-oriented representation distillation)

Eq. (16) bi-level objective:
    min_theta L_BPR(theta, delta*)  s.t.  delta* = argmin_delta L_info(theta, delta)

Exact nested bi-level optimization (a full inner argmin every outer step) is
computationally prohibitive, so -- as is standard practice for this style of
formulation -- we APPROXIMATE it via alternating gradient steps each epoch:
one BPR mini-batch update (theta), then one contrastive mini-batch update
(theta AND delta jointly, since Eq. 10/13's gradient naturally flows back to
both the real item embeddings and the noise table -- see contrastive_loss.py).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from models.mf import MatrixFactorization, _build_user_positive_items, _sample_negatives
from fairness.cross_group_noise import (
    CrossGroupNoise, build_group_positive_sets, sample_cross_group_items,
)
from fairness.contrastive_loss import compute_loss_and_grad_single_user


def train_fairir(
    train_df: pd.DataFrame,
    user_attrs: pd.DataFrame,
    n_users: int,
    n_items: int,
    group_col: str = "gender_code",
    n_factors: int = 64,
    n_epochs: int = 15,
    bpr_batch_size: int = 1024,
    bpr_lr: float = 0.05,
    bpr_reg: float = 0.01,
    contrastive_batch_size: int = 256,
    contrastive_lr: float = 0.05,
    n_negatives: int = 5,
    delta: float = 0.1,
    alpha: float = 0.5,
    tau: float = 0.7,
    n_pos_cap: int = 20,
    seed: int = 0,
    verbose: bool = True,
) -> tuple:
    """
    Returns (model, noise_module) -- model.user_emb/item_emb are theta,
    noise_module.noise is delta, both updated jointly per the bi-level
    approximation described above.
    """
    rng = np.random.default_rng(seed)
    model = MatrixFactorization(n_users, n_items, n_factors=n_factors, seed=seed)
    noise_module = CrossGroupNoise(n_items, n_factors, delta=delta, seed=seed)

    user_positive_items = _build_user_positive_items(train_df, n_users)
    group_positive_sets = build_group_positive_sets(train_df, user_attrs, group_col)

    user_group_map = np.full(n_users, -1)
    for u, g in zip(user_attrs["user_id"].values, user_attrs[group_col].values):
        user_group_map[u] = g

    interactions_u = train_df["user_id"].values
    interactions_i = train_df["item_id"].values
    n_interactions = len(interactions_u)
    all_user_ids = np.arange(n_users)

    for epoch in range(n_epochs):
        perm = rng.permutation(n_interactions)
        bpr_loss_sum = 0.0

        # --- Step 1: standard BPR mini-batch updates (outer objective, theta) ---
        for start in range(0, n_interactions, bpr_batch_size):
            batch_idx = perm[start : start + bpr_batch_size]
            u = interactions_u[batch_idx]
            i_pos = interactions_i[batch_idx]
            i_neg = _sample_negatives(u, user_positive_items, n_items, rng)

            p_u = model.user_emb[u]
            q_pos = model.item_emb[i_pos]
            q_neg = model.item_emb[i_neg]

            x_uij = np.sum(p_u * (q_pos - q_neg), axis=1)
            sigmoid_neg = 1.0 / (1.0 + np.exp(x_uij))
            bpr_loss_sum += (-np.log(1.0 / (1.0 + np.exp(-x_uij)) + 1e-12)).sum()

            grad_common = sigmoid_neg[:, None]
            grad_p_u = -grad_common * (q_pos - q_neg) + bpr_reg * p_u
            grad_q_pos = -grad_common * p_u + bpr_reg * q_pos
            grad_q_neg = grad_common * p_u + bpr_reg * q_neg

            np.add.at(model.user_emb, u, -bpr_lr * grad_p_u)
            np.add.at(model.item_emb, i_pos, -bpr_lr * grad_q_pos)
            np.add.at(model.item_emb, i_neg, -bpr_lr * grad_q_neg)

        # --- Step 2: contrastive mini-batch updates (inner objective, delta + theta) ---
        contrastive_loss_sum = 0.0
        n_contrastive_users = 0
        user_perm = rng.permutation(all_user_ids)

        for start in range(0, n_users, contrastive_batch_size):
            batch_users = user_perm[start : start + contrastive_batch_size]

            for u in batch_users:
                real_items = user_positive_items[u]
                if len(real_items) == 0:
                    continue
                own_group = user_group_map[u]
                if own_group < 0:
                    continue

                # Cap the number of positive items used per user per epoch,
                # regardless of that user's total interaction count -- using
                # ALL of a heavy user's positives (some ML-1M users have
                # hundreds/thousands of ratings) made the summed loss/gradient
                # scale with activity level and caused training to diverge on
                # real data. n_pos_cap keeps this bounded and comparable
                # across users of very different activity levels.
                n_total = min(len(real_items), n_pos_cap)
                n_noised = max(1, round(alpha * n_total))
                n_real = max(1, n_total - n_noised)

                real_subset = rng.choice(real_items, size=min(n_real, len(real_items)), replace=False)
                cross_items = sample_cross_group_items(own_group, group_positive_sets, n_noised, rng)
                if len(cross_items) == 0:
                    continue

                real_pos_embs = [model.item_emb[i] for i in real_subset]
                noised_pos_embs = [
                    noise_module.get_noised_item_emb(model.item_emb, np.array([i]))[0]
                    for i in cross_items
                ]
                pos_embs = real_pos_embs + noised_pos_embs
                n_pos_used = len(pos_embs)

                neg_ids = _sample_negatives(
                    np.full(n_negatives, u), user_positive_items, n_items, rng
                )
                neg_embs = [model.item_emb[j] for j in neg_ids]

                loss, grad_u, grad_pos, grad_neg = compute_loss_and_grad_single_user(
                    model.user_emb[u], pos_embs, neg_embs, tau
                )
                contrastive_loss_sum += loss
                n_contrastive_users += 1

                # Average (not sum) the update by n_pos_used, so a user with
                # many positive items doesn't produce a proportionally larger
                # gradient step than a user with few -- this is what kept
                # training stable once n_pos_cap alone wasn't enough.
                scale = contrastive_lr / n_pos_used
                model.user_emb[u] -= scale * grad_u
                for idx, i in enumerate(real_subset):
                    model.item_emb[i] -= scale * grad_pos[idx]
                for idx, i in enumerate(cross_items):
                    noise_module.noise[i] -= scale * grad_pos[len(real_subset) + idx]
                for idx, j in enumerate(neg_ids):
                    model.item_emb[j] -= scale * grad_neg[idx]

        if verbose:
            avg_bpr = bpr_loss_sum / n_interactions
            avg_contrastive = contrastive_loss_sum / max(1, n_contrastive_users)
            print(f"epoch {epoch + 1}/{n_epochs}  bpr_loss={avg_bpr:.4f}  "
                  f"contrastive_loss={avg_contrastive:.4f}")

    return model, noise_module


if __name__ == "__main__":
    # Quick end-to-end check on synthetic data (fast, no real files needed).
    rng = np.random.default_rng(0)
    n_users, n_items = 60, 40

    train_df = pd.DataFrame({
        "user_id": rng.integers(0, n_users, size=800),
        "item_id": rng.integers(0, n_items, size=800),
    }).drop_duplicates().reset_index(drop=True)

    user_attrs = pd.DataFrame({
        "user_id": np.arange(n_users),
        "gender_code": rng.integers(0, 2, size=n_users),
    })

    model, noise_module = train_fairir(
        train_df, user_attrs, n_users, n_items,
        n_factors=16, n_epochs=5, seed=0,
    )
    print("\nDone. user_emb shape:", model.user_emb.shape)
    print("noise shape:", noise_module.noise.shape)
    print("noise mean abs value:", np.abs(noise_module.noise).mean())