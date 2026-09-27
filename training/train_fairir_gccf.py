"""
train_fairir_gccf.py
-----------------------
FairIR (cross-group noise + contrastive distillation) trained on top of the
GCCF backbone instead of plain MF -- to test whether FairIR's own documented
degradation (Section 4.4: "message passing... is equivalent to a low-pass
filter and attenuates high-frequency, group-specific perturbations") shows
up in OUR reimplementation too, before building any fix for it.

Design: each epoch, GCCF's forward() propagates the CURRENT base embeddings
once; BPR mini-batches AND the contrastive mini-batches both accumulate
their gradients into the same grad_final_user/grad_final_item buffers
(w.r.t. the FINAL/propagated embeddings); ONE backward() call per epoch then
routes the combined gradient back through the graph layers to the base
embeddings. The cross-group noise table is independent of the graph (it's a
per-item learnable perturbation added directly to the final propagated item
embedding when constructing noised cross-group samples), so it's updated
the same way as in training/train_fairir.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from models.gccf import GCCF
from models.mf import _build_user_positive_items, _sample_negatives
from fairness.cross_group_noise import (
    CrossGroupNoise, build_group_positive_sets, sample_cross_group_items,
)
from fairness.contrastive_loss import compute_loss_and_grad_single_user


def train_fairir_gccf(
    train_df: pd.DataFrame,
    user_attrs: pd.DataFrame,
    n_users: int,
    n_items: int,
    group_col: str = "gender_code",
    n_factors: int = 64,
    n_layers: int = 2,
    n_epochs: int = 15,
    bpr_batch_size: int = 1024,
    bpr_lr: float = 0.05,
    bpr_reg: float = 0.01,
    bpr_accumulation_steps: int = 10,
    contrastive_batch_size: int = 256,
    contrastive_lr: float = 0.05,
    contrastive_accumulation_steps: int = 60,
    n_negatives: int = 5,
    n_pos_cap: int = 20,
    delta: float = 0.1,
    alpha: float = 0.5,
    tau: float = 0.7,
    seed: int = 0,
    verbose: bool = True,
) -> tuple:
    rng = np.random.default_rng(seed)
    model = GCCF(n_users, n_items, n_factors, train_df, n_layers=n_layers, seed=seed)
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
        model.forward()  # propagate current base embeddings once for this epoch

        # --- Phase 1: BPR, periodically flushed (same fix as models/gccf.py's
        # train_gccf_bpr -- accumulating the WHOLE epoch into one update made
        # the effective step hundreds of times too large and diverged; a
        # single per-epoch AVERAGE was stable but far too slow to learn) ---
        perm = rng.permutation(n_interactions)
        bpr_loss_sum = 0.0
        grad_final_user = np.zeros_like(model.user_emb)
        grad_final_item = np.zeros_like(model.item_emb)
        batches_in_group = 0

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

            np.add.at(grad_final_user, u, grad_p_u)
            np.add.at(grad_final_item, i_pos, grad_q_pos)
            np.add.at(grad_final_item, i_neg, grad_q_neg)
            batches_in_group += 1

            if batches_in_group >= bpr_accumulation_steps:
                grad_final_user /= batches_in_group
                grad_final_item /= batches_in_group
                gu, gi = model.backward(grad_final_user, grad_final_item)
                model.user_emb_base -= bpr_lr * gu
                model.item_emb_base -= bpr_lr * gi
                grad_final_user[:] = 0
                grad_final_item[:] = 0
                batches_in_group = 0

        if batches_in_group > 0:
            grad_final_user /= batches_in_group
            grad_final_item /= batches_in_group
            gu, gi = model.backward(grad_final_user, grad_final_item)
            model.user_emb_base -= bpr_lr * gu
            model.item_emb_base -= bpr_lr * gi

        model.forward()  # refresh propagated embeddings before the contrastive phase

        # --- Phase 2: contrastive, also periodically flushed (same fix --
        # summing thousands of individual per-user contrastive gradients
        # into one epoch-end update was the direct cause of the observed
        # contrastive_loss exploding to astronomical values) ---
        contrastive_loss_sum = 0.0
        n_contrastive_users = 0
        user_perm = rng.permutation(all_user_ids)

        grad_final_user = np.zeros_like(model.user_emb)
        grad_final_item = np.zeros_like(model.item_emb)
        users_in_group = 0

        for u in user_perm:
            real_items = user_positive_items[u]
            if len(real_items) == 0:
                continue
            own_group = user_group_map[u]
            if own_group < 0:
                continue

            n_total = min(len(real_items), n_pos_cap)
            n_noised = max(1, round(alpha * n_total))
            n_real = max(1, n_total - n_noised)

            real_subset = rng.choice(real_items, size=min(n_real, len(real_items)), replace=False)
            cross_items = sample_cross_group_items(own_group, group_positive_sets, n_noised, rng)
            if len(cross_items) == 0:
                continue

            real_pos_embs = [model.item_emb[i] for i in real_subset]
            noised_pos_embs = [model.item_emb[i] + noise_module.noise[i] for i in cross_items]
            pos_embs = real_pos_embs + noised_pos_embs
            n_pos_used = len(pos_embs)

            neg_ids = _sample_negatives(np.full(n_negatives, u), user_positive_items, n_items, rng)
            neg_embs = [model.item_emb[j] for j in neg_ids]

            loss, grad_u, grad_pos, grad_neg = compute_loss_and_grad_single_user(
                model.user_emb[u], pos_embs, neg_embs, tau
            )
            contrastive_loss_sum += loss
            n_contrastive_users += 1

            per_user_scale = 1.0 / n_pos_used
            grad_final_user[u] += per_user_scale * grad_u
            for idx, i in enumerate(real_subset):
                grad_final_item[i] += per_user_scale * grad_pos[idx]
            for idx, i in enumerate(cross_items):
                noise_module.noise[i] -= contrastive_lr * per_user_scale * grad_pos[len(real_subset) + idx]
            for idx, j in enumerate(neg_ids):
                grad_final_item[j] += per_user_scale * grad_neg[idx]
            users_in_group += 1

            if users_in_group >= contrastive_accumulation_steps:
                grad_final_user /= users_in_group
                grad_final_item /= users_in_group
                gu, gi = model.backward(grad_final_user, grad_final_item)
                model.user_emb_base -= contrastive_lr * gu
                model.item_emb_base -= contrastive_lr * gi
                grad_final_user[:] = 0
                grad_final_item[:] = 0
                users_in_group = 0

        if users_in_group > 0:
            grad_final_user /= users_in_group
            grad_final_item /= users_in_group
            gu, gi = model.backward(grad_final_user, grad_final_item)
            model.user_emb_base -= contrastive_lr * gu
            model.item_emb_base -= contrastive_lr * gi

        if verbose:
            avg_bpr = bpr_loss_sum / n_interactions
            avg_contrastive = contrastive_loss_sum / max(1, n_contrastive_users)
            print(f"epoch {epoch + 1}/{n_epochs}  bpr_loss={avg_bpr:.4f}  contrastive_loss={avg_contrastive:.4f}")

    model.forward()  # final propagation with fully-trained base embeddings
    return model, noise_module


if __name__ == "__main__":
    # Quick end-to-end check on synthetic data.
    rng = np.random.default_rng(0)
    n_users, n_items = 300, 200

    train_df = pd.DataFrame({
        "user_id": rng.integers(0, n_users, size=8000),
        "item_id": rng.integers(0, n_items, size=8000),
    }).drop_duplicates().reset_index(drop=True)

    user_attrs = pd.DataFrame({
        "user_id": np.arange(n_users),
        "gender_code": rng.integers(0, 2, size=n_users),
    })

    model, noise_module = train_fairir_gccf(
        train_df, user_attrs, n_users, n_items,
        n_factors=32, n_layers=2, n_epochs=10, seed=0,
    )
    print("\nDone. any NaN:", np.isnan(model.user_emb).any() or np.isnan(model.item_emb).any())