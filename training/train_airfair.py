"""
train_airfair.py
------------------
AIR-Fair training loop: identical to training/train_fairir.py, EXCEPT the
global fixed (alpha, delta) are replaced by per-user (alpha_u, delta_u)
computed via fairness/instance_gate.py -- the Instance-Adaptive Gate
component of AIR-Fair.

The per-user gate is recomputed once per epoch (using the model's CURRENT
embeddings, train data only -- no test leakage), so correction strength
tracks how well each user is currently being served relative to their
dominant group, and adapts as training progresses.

Everything else (BPR step, cross-group noise pool, contrastive loss) is
unchanged from train_fairir.py -- only alpha/delta become per-user instead
of global constants.
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
from fairness.instance_gate import (
    compute_user_quality_proxy, compute_gate_values, get_adaptive_alpha_delta,
)


def train_airfair(
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
    n_pos_cap: int = 20,
    delta_base: float = 0.1,
    alpha_base: float = 0.5,  # only used as the FALLBACK/init value before the first gate computation
    alpha_min: float = 0.1,
    alpha_max: float = 0.9,
    delta_min_ratio: float = 0.5,
    delta_max_ratio: float = 2.0,
    invert_gate: bool = False,
    randomize_gate: bool = False,
    tau: float = 0.7,
    gate_recompute_every: int = 1,  # epochs between gate recomputation
    seed: int = 0,
    verbose: bool = True,
) -> tuple:
    """
    Returns (model, noise_module, last_gate_info) -- last_gate_info is a dict
    with the final epoch's {alpha_u, delta_u, g_values, dominant_group} for
    inspection/logging.
    """
    rng = np.random.default_rng(seed)
    model = MatrixFactorization(n_users, n_items, n_factors=n_factors, seed=seed)
    noise_module = CrossGroupNoise(n_items, n_factors, delta=delta_base, seed=seed)

    user_positive_items = _build_user_positive_items(train_df, n_users)
    group_positive_sets = build_group_positive_sets(train_df, user_attrs, group_col)

    user_group_map = np.full(n_users, -1)
    for u, g in zip(user_attrs["user_id"].values, user_attrs[group_col].values):
        user_group_map[u] = g

    interactions_u = train_df["user_id"].values
    interactions_i = train_df["item_id"].values
    n_interactions = len(interactions_u)
    all_user_ids = np.arange(n_users)

    alpha_u = np.full(n_users, alpha_base)
    delta_u = np.full(n_users, delta_base)
    last_gate_info = {}

    for epoch in range(n_epochs):
        # --- Recompute the instance-adaptive gate periodically, using the
        # model's CURRENT embeddings and TRAIN data only. ---
        if epoch % gate_recompute_every == 0:
            quality = compute_user_quality_proxy(
                model, train_df, n_users, n_items, n_samples_per_user=50, seed=seed + epoch
            )
            g_values, dominant_group, dominant_quality = compute_gate_values(quality, user_group_map)
            if randomize_gate:
                # ABLATION: shuffle g(u) across users before mapping to alpha_u/delta_u.
                # This preserves the EXACT SAME population distribution of correction
                # strengths (same alpha_min/alpha_max range, same overall variance) but
                # severs the link between a user's TRUE need and the correction they
                # receive -- isolating whether AIR-Fair's benefit comes from targeted
                # correction itself, or merely from having ANY per-user variability.
                g_values = rng.permutation(g_values)
            alpha_u, delta_u = get_adaptive_alpha_delta(
                g_values, delta_base=delta_base, alpha_min=alpha_min, alpha_max=alpha_max,
                delta_min_ratio=delta_min_ratio, delta_max_ratio=delta_max_ratio,
                invert=invert_gate,
            )
            last_gate_info = {
                "alpha_u": alpha_u, "delta_u": delta_u, "g_values": g_values,
                "dominant_group": dominant_group, "dominant_quality": dominant_quality,
            }

        perm = rng.permutation(n_interactions)
        bpr_loss_sum = 0.0

        # --- Step 1: standard BPR mini-batch updates (unchanged from train_fairir.py) ---
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

        # --- Step 2: contrastive mini-batch updates, now with PER-USER alpha_u/delta_u ---
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

                this_alpha = alpha_u[u]
                this_delta_ratio = delta_u[u] / delta_base  # scales the shared learned noise direction

                n_total = min(len(real_items), n_pos_cap)
                n_noised = max(1, round(this_alpha * n_total))
                n_real = max(1, n_total - n_noised)

                real_subset = rng.choice(real_items, size=min(n_real, len(real_items)), replace=False)
                cross_items = sample_cross_group_items(own_group, group_positive_sets, n_noised, rng)
                if len(cross_items) == 0:
                    continue

                real_pos_embs = [model.item_emb[i] for i in real_subset]
                noised_pos_embs = [
                    model.item_emb[i] + this_delta_ratio * noise_module.noise[i]
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

                scale = contrastive_lr / n_pos_used
                model.user_emb[u] -= scale * grad_u
                for idx, i in enumerate(real_subset):
                    model.item_emb[i] -= scale * grad_pos[idx]
                for idx, i in enumerate(cross_items):
                    # Gradient w.r.t. the noised embedding flows straight to the
                    # shared noise[i] parameter, scaled by this user's delta ratio
                    # (chain rule: d(noised_emb)/d(noise[i]) = this_delta_ratio).
                    noise_module.noise[i] -= scale * this_delta_ratio * grad_pos[len(real_subset) + idx]
                for idx, j in enumerate(neg_ids):
                    model.item_emb[j] -= scale * grad_neg[idx]

        if verbose:
            avg_bpr = bpr_loss_sum / n_interactions
            avg_contrastive = contrastive_loss_sum / max(1, n_contrastive_users)
            print(f"epoch {epoch + 1}/{n_epochs}  bpr_loss={avg_bpr:.4f}  "
                  f"contrastive_loss={avg_contrastive:.4f}  "
                  f"mean_alpha_u={alpha_u.mean():.3f}  mean_delta_u={delta_u.mean():.4f}")

    return model, noise_module, last_gate_info


if __name__ == "__main__":
    # Quick end-to-end check on synthetic data with a manufactured group
    # quality gap, mirroring instance_gate.py's own test scenario.
    rng = np.random.default_rng(0)
    n_users, n_items = 60, 40

    train_df = pd.DataFrame({
        "user_id": rng.integers(0, n_users, size=800),
        "item_id": rng.integers(0, n_items, size=800),
    }).drop_duplicates().reset_index(drop=True)

    user_attrs = pd.DataFrame({
        "user_id": np.arange(n_users),
        "gender_code": ([0] * 45 + [1] * 15),  # imbalanced, like real ML-1M
    })

    model, noise_module, gate_info = train_airfair(
        train_df, user_attrs, n_users, n_items,
        n_factors=16, n_epochs=5, seed=0,
    )
    print("\nDone. Final gate info:")
    print(f"Dominant group: {gate_info['dominant_group']} (quality={gate_info['dominant_quality']:.4f})")
    print(f"Group 0 mean alpha_u: {gate_info['alpha_u'][user_attrs['gender_code']==0].mean():.4f}")
    print(f"Group 1 mean alpha_u: {gate_info['alpha_u'][user_attrs['gender_code']==1].mean():.4f}")