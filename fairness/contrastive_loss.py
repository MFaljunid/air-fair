"""
contrastive_loss.py
---------------------
FairIR's "Fairness-Oriented Representation Distillation" (Shi et al., KBS
2026, Section 3.3, Equations 7-14). InfoNCE-style contrastive loss that
pulls a user's "final" positive-item set (real items + noised cross-group
items, mixed via balance parameter alpha) toward the user embedding, while
pushing negative (non-interacted) items away.

    Eq. (11): sim(v_u, v_i) = exp(v_u^T v_i / tau)

    Eq. (10)/(13): for each positive item i in the user's final positive set,
        L_i = -log( sim(u,i) / (sim(u,i) + sum_{j in negs} sim(u,j)) )
        L_info = E_u[ sum_i L_i ]

    Eq. (12): final positive set mixes real and noised cross-group items,
        controlled by balance parameter alpha in [0,1] (fraction that is
        noised/cross-group).

This module works at the EMBEDDING level (pure NumPy, matching models/mf.py's
manual-gradient style): given a user embedding, a list of "final positive"
item embeddings, and a list of negative item embeddings, it returns the loss
value AND analytic gradients w.r.t. every embedding involved. The caller
(training/train_fairir.py) is responsible for routing each gradient back to
the right place -- models/mf.py's item_emb table for real items, or
cross_group_noise.py's noise table for noised cross-group items.
"""

import numpy as np


def similarity(u_emb: np.ndarray, i_emb: np.ndarray, tau: float) -> float:
    """
    Eq. (11): sim(v_u, v_i) = exp(v_u^T v_i / tau).
    NOTE: kept for reference / external callers, but
    compute_loss_and_grad_single_user() below does NOT call this directly --
    it computes raw scores (u.v/tau) and combines them via a numerically
    stable log-sum-exp softmax instead of ever exponentiating a raw score,
    since exp() of a large dot product overflows in float64 once embeddings
    grow during training (observed empirically on real ML-1M data).
    """
    return np.exp(np.dot(u_emb, i_emb) / tau)


def compute_loss_and_grad_single_user(
    u_emb: np.ndarray,
    pos_embs: list,   # list of (n_factors,) arrays -- the user's "final" positive items
    neg_embs: list,   # list of (n_factors,) arrays -- this user's negative samples
    tau: float,
) -> tuple:
    """
    Eq. (10)/(13) for ONE user: each positive item is scored against the
    SAME shared set of negatives for that user. Returns:
        loss        : scalar, sum over all positive items for this user
        grad_u      : (n_factors,) gradient w.r.t. u_emb
        grad_pos    : list of (n_factors,) gradients, one per pos_embs entry
        grad_neg    : list of (n_factors,) gradients, one per neg_embs entry
                      (accumulated across all positive-item terms, since the
                      same negatives are reused for every positive)

    Numerically stable: works with raw scores (u.v/tau) and a shifted
    softmax (subtract the max score before exponentiating) instead of
    calling similarity()/exp() directly on raw scores, which overflows once
    embeddings grow large during training. Mathematically identical to the
    naive exp(...)/sum(exp(...)) formulation -- only the numerics differ.
    """
    n_factors = u_emb.shape[0]
    grad_u = np.zeros(n_factors)
    grad_pos = [np.zeros(n_factors) for _ in pos_embs]
    grad_neg = [np.zeros(n_factors) for _ in neg_embs]
    total_loss = 0.0

    neg_scores = [np.dot(u_emb, n) / tau for n in neg_embs]

    for p_idx, p_emb in enumerate(pos_embs):
        pos_score = np.dot(u_emb, p_emb) / tau
        all_scores = [pos_score] + neg_scores
        max_score = max(all_scores)

        shifted = [s - max_score for s in all_scores]
        exp_shifted = [np.exp(s) for s in shifted]
        sum_exp = sum(exp_shifted)

        # loss = -shifted_pos + log(sum_exp)  (== -log(pos_sim/denom), stable)
        total_loss += -shifted[0] + np.log(sum_exp + 1e-12)

        prob_pos = exp_shifted[0] / sum_exp
        dL_dscore_pos = (prob_pos - 1.0)
        grad_u += dL_dscore_pos * (p_emb / tau)
        grad_pos[p_idx] += dL_dscore_pos * (u_emb / tau)

        for n_idx, n_emb in enumerate(neg_embs):
            prob_neg = exp_shifted[n_idx + 1] / sum_exp
            dL_dscore_neg = prob_neg
            grad_u += dL_dscore_neg * (n_emb / tau)
            grad_neg[n_idx] += dL_dscore_neg * (u_emb / tau)

    return total_loss, grad_u, grad_pos, grad_neg


def mix_positive_set(real_item_embs: list, noised_item_embs: list, alpha: float) -> list:
    """
    Eq. (12): final positive set = (1-alpha) fraction real items + alpha
    fraction noised cross-group items. Given the already-selected real and
    noised embedding lists (selection/counting is the caller's job, e.g.
    n_real = round((1-alpha)*n_total), n_noised = round(alpha*n_total)),
    this just concatenates them into the single list compute_loss_and_grad_single_user expects.
    """
    return list(real_item_embs) + list(noised_item_embs)


if __name__ == "__main__":
    # Gradient check: compare the analytic gradient above against a numerical
    # (finite-difference) gradient, to verify the hand-derived formulas.
    rng = np.random.default_rng(0)
    n_factors = 6
    tau = 0.7

    u_emb = rng.normal(0, 1, size=n_factors)
    pos_embs = [rng.normal(0, 1, size=n_factors) for _ in range(2)]
    neg_embs = [rng.normal(0, 1, size=n_factors) for _ in range(4)]

    loss, grad_u, grad_pos, grad_neg = compute_loss_and_grad_single_user(
        u_emb, pos_embs, neg_embs, tau
    )
    print(f"Analytic loss: {loss:.6f}")

    def loss_fn(u, pos_list, neg_list):
        l, *_ = compute_loss_and_grad_single_user(u, pos_list, neg_list, tau)
        return l

    eps = 1e-5

    def numerical_grad_vector(perturb_fn, dim):
        grad = np.zeros(dim)
        for d in range(dim):
            plus = perturb_fn(d, eps)
            minus = perturb_fn(d, -eps)
            grad[d] = (plus - minus) / (2 * eps)
        return grad

    # Check grad w.r.t. u_emb
    def perturb_u(d, delta):
        u2 = u_emb.copy()
        u2[d] += delta
        return loss_fn(u2, pos_embs, neg_embs)

    num_grad_u = numerical_grad_vector(perturb_u, n_factors)
    print(f"\ngrad_u analytic:  {grad_u}")
    print(f"grad_u numerical: {num_grad_u}")
    print(f"max abs diff: {np.abs(grad_u - num_grad_u).max():.8f}")
    assert np.allclose(grad_u, num_grad_u, atol=1e-4), "grad_u mismatch!"

    # Check grad w.r.t. first positive item embedding
    def perturb_pos0(d, delta):
        p2 = [e.copy() for e in pos_embs]
        p2[0][d] += delta
        return loss_fn(u_emb, p2, neg_embs)

    num_grad_pos0 = numerical_grad_vector(perturb_pos0, n_factors)
    print(f"\ngrad_pos[0] analytic:  {grad_pos[0]}")
    print(f"grad_pos[0] numerical: {num_grad_pos0}")
    print(f"max abs diff: {np.abs(grad_pos[0] - num_grad_pos0).max():.8f}")
    assert np.allclose(grad_pos[0], num_grad_pos0, atol=1e-4), "grad_pos mismatch!"

    # Check grad w.r.t. first negative item embedding
    def perturb_neg0(d, delta):
        n2 = [e.copy() for e in neg_embs]
        n2[0][d] += delta
        return loss_fn(u_emb, pos_embs, n2)

    num_grad_neg0 = numerical_grad_vector(perturb_neg0, n_factors)
    print(f"\ngrad_neg[0] analytic:  {grad_neg[0]}")
    print(f"grad_neg[0] numerical: {num_grad_neg0}")
    print(f"max abs diff: {np.abs(grad_neg[0] - num_grad_neg0).max():.8f}")
    assert np.allclose(grad_neg[0], num_grad_neg0, atol=1e-4), "grad_neg mismatch!"

    print("\nOK: all analytic gradients match numerical gradients.")