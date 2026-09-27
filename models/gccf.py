"""
gccf.py
--------
LR-GCCF backbone (Chen et al., AAAI 2020, "Revisiting Graph based
Collaborative Filtering: A Linear Residual Graph Convolutional Network
Approach") -- the GNN backbone FairIR itself uses as its second backbone
(baseline "GCCF" in their Table 3-5), and where FairIR's own Section 4.4
documents that message-passing acts as a low-pass filter, attenuating the
group-specific fairness signal.

Purely linear graph propagation (no activation, no per-layer weight
matrix), which makes it possible to derive an exact, hand-rolled backward
pass -- verified below against numerical (finite-difference) gradients,
the same rigor used for fairness/contrastive_loss.py.

Design:
    E_user^(0), E_item^(0)  = learnable BASE embeddings (the only trainable
                               parameters -- everything else is a fixed
                               linear function of them).
    E_user^(k) = norm_UI   @ E_item^(k-1)   for k = 1..K
    E_item^(k) = norm_UI.T @ E_user^(k-1)   for k = 1..K
    Final_user = sum_{k=0}^{K} E_user^(k)   (residual sum, per LR-GCCF)
    Final_item = sum_{k=0}^{K} E_item^(k)

norm_UI[u, i] = 1 / sqrt(deg(u) * deg(i))  for observed (u, i) train pairs,
0 otherwise -- the bipartite-graph symmetric normalization used by
LightGCN/LR-GCCF-style models.
"""

import numpy as np
import scipy.sparse as sp
import pandas as pd


def build_normalized_adjacency(train_df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    """norm_UI[u,i] = 1/sqrt(deg(u)*deg(i)) for observed interactions, sparse (n_users x n_items)."""
    user_ids = train_df["user_id"].values
    item_ids = train_df["item_id"].values

    user_deg = np.bincount(user_ids, minlength=n_users).astype(float)
    item_deg = np.bincount(item_ids, minlength=n_items).astype(float)
    user_deg[user_deg == 0] = 1.0  # avoid div-by-zero for isolated nodes (shouldn't occur post-LOO, but safe)
    item_deg[item_deg == 0] = 1.0

    weights = 1.0 / np.sqrt(user_deg[user_ids] * item_deg[item_ids])
    norm_UI = sp.csr_matrix((weights, (user_ids, item_ids)), shape=(n_users, n_items))
    return norm_UI


class GCCF:
    def __init__(
        self, n_users: int, n_items: int, n_factors: int, train_df: pd.DataFrame,
        n_layers: int = 2, seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        scale = 0.01
        self.n_factors = n_factors
        self.n_layers = n_layers
        self.n_users = n_users
        self.n_items = n_items

        self.user_emb_base = rng.normal(0, scale, size=(n_users, n_factors))
        self.item_emb_base = rng.normal(0, scale, size=(n_items, n_factors))

        self.norm_UI = build_normalized_adjacency(train_df, n_users, n_items)
        self.norm_IU = self.norm_UI.T.tocsr()

        # Populated by forward(); model.user_emb/model.item_emb are what
        # eval/metrics.py and predict() read, matching models/mf.py's interface.
        self.user_emb = self.user_emb_base.copy()
        self.item_emb = self.item_emb_base.copy()
        self._user_layers = None
        self._item_layers = None

    def forward(self):
        """Propagate the CURRENT base embeddings through n_layers graph-conv
        steps and cache every intermediate layer (needed for backward()).
        Sets self.user_emb / self.item_emb to the final (residual-summed)
        embeddings, so predict() and eval/metrics.py work unchanged."""
        user_layers = [self.user_emb_base]
        item_layers = [self.item_emb_base]
        for k in range(self.n_layers):
            next_user = self.norm_UI @ item_layers[-1]
            next_item = self.norm_IU @ user_layers[-1]
            user_layers.append(next_user)
            item_layers.append(next_item)

        self._user_layers = user_layers
        self._item_layers = item_layers
        self.user_emb = sum(user_layers)
        self.item_emb = sum(item_layers)
        return self.user_emb, self.item_emb

    def backward(self, grad_final_user: np.ndarray, grad_final_item: np.ndarray) -> tuple:
        """
        Exact backward pass through the residual sum + linear propagation,
        back to the BASE embeddings (the only learnable parameters).
        Returns (grad_user_base, grad_item_base), same shape as the base
        embedding tables.
        """
        K = self.n_layers
        grad_user_base = np.zeros_like(self.user_emb_base)
        grad_item_base = np.zeros_like(self.item_emb_base)

        # Contribution of each of the K+1 terms in Final_user = sum_k E_user^(k)
        for k in range(K + 1):
            cur_grad = grad_final_user
            is_user_side = True
            for _ in range(k):
                if is_user_side:
                    cur_grad = self.norm_IU @ cur_grad  # d E_user^(j)/d E_item^(j-1) backward = norm_UI.T = norm_IU
                    is_user_side = False
                else:
                    cur_grad = self.norm_UI @ cur_grad  # d E_item^(j)/d E_user^(j-1) backward = norm_UI
                    is_user_side = True
            if is_user_side:
                grad_user_base += cur_grad
            else:
                grad_item_base += cur_grad

        # Contribution of each of the K+1 terms in Final_item = sum_k E_item^(k)
        for k in range(K + 1):
            cur_grad = grad_final_item
            is_item_side = True
            for _ in range(k):
                if is_item_side:
                    cur_grad = self.norm_UI @ cur_grad  # d E_item^(j)/d E_user^(j-1) backward = norm_UI
                    is_item_side = False
                else:
                    cur_grad = self.norm_IU @ cur_grad
                    is_item_side = True
            if is_item_side:
                grad_item_base += cur_grad
            else:
                grad_user_base += cur_grad

        return grad_user_base, grad_item_base

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        return np.sum(self.user_emb[user_ids] * self.item_emb[item_ids], axis=1)


def train_gccf_bpr(
    train_df: pd.DataFrame, n_users: int, n_items: int,
    n_factors: int = 64, n_layers: int = 2, n_epochs: int = 15,
    batch_size: int = 1024, lr: float = 0.05, reg: float = 0.01,
    accumulation_steps: int = 10, seed: int = 0,
    verbose: bool = True,
) -> "GCCF":
    """
    Trains GCCF with BPR loss. Graph propagation (forward()) is recomputed
    ONCE per epoch (standard practice for GCN-style recommenders -- the
    expensive sparse propagation is shared across that epoch's mini-batches;
    gradients from ALL mini-batches are accumulated and applied to the base
    embeddings via ONE backward() call at the end of the epoch).
    """
    rng = np.random.default_rng(seed)
    model = GCCF(n_users, n_items, n_factors, train_df, n_layers=n_layers, seed=seed)

    from models.mf import _build_user_positive_items, _sample_negatives
    user_positive_items = _build_user_positive_items(train_df, n_users)

    interactions_u = train_df["user_id"].values
    interactions_i = train_df["item_id"].values
    n_interactions = len(interactions_u)

    for epoch in range(n_epochs):
        model.forward()  # propagate with the current base embeddings (once per epoch)

        perm = rng.permutation(n_interactions)
        epoch_loss = 0.0

        grad_final_user = np.zeros_like(model.user_emb)
        grad_final_item = np.zeros_like(model.item_emb)
        batches_in_group = 0

        for start in range(0, n_interactions, batch_size):
            batch_idx = perm[start : start + batch_size]
            u = interactions_u[batch_idx]
            i_pos = interactions_i[batch_idx]
            i_neg = _sample_negatives(u, user_positive_items, n_items, rng)

            p_u = model.user_emb[u]
            q_pos = model.item_emb[i_pos]
            q_neg = model.item_emb[i_neg]

            x_uij = np.sum(p_u * (q_pos - q_neg), axis=1)
            sigmoid_neg = 1.0 / (1.0 + np.exp(x_uij))
            epoch_loss += (-np.log(1.0 / (1.0 + np.exp(-x_uij)) + 1e-12)).sum()

            grad_common = sigmoid_neg[:, None]
            grad_p_u = -grad_common * (q_pos - q_neg) + reg * p_u
            grad_q_pos = -grad_common * p_u + reg * q_pos
            grad_q_neg = grad_common * p_u + reg * q_neg

            np.add.at(grad_final_user, u, grad_p_u)
            np.add.at(grad_final_item, i_pos, grad_q_pos)
            np.add.at(grad_final_item, i_neg, grad_q_neg)
            batches_in_group += 1

            # Apply an update every `accumulation_steps` mini-batches -- a
            # middle ground between updating every single batch (correct
            # step size, but backward()'s Python-level overhead made ~550
            # calls/epoch too slow: >100s/epoch) and accumulating the WHOLE
            # epoch into one update (fast, but averaging over ~550 batches
            # made the step size too small to learn anything at all --
            # observed empirically: loss stayed frozen at ln(2)=0.6931).
            if batches_in_group >= accumulation_steps:
                grad_final_user /= batches_in_group
                grad_final_item /= batches_in_group
                grad_user_base, grad_item_base = model.backward(grad_final_user, grad_final_item)
                model.user_emb_base -= lr * grad_user_base
                model.item_emb_base -= lr * grad_item_base
                grad_final_user[:] = 0
                grad_final_item[:] = 0
                batches_in_group = 0

        # Flush any remaining batches at the end of the epoch.
        if batches_in_group > 0:
            grad_final_user /= batches_in_group
            grad_final_item /= batches_in_group
            grad_user_base, grad_item_base = model.backward(grad_final_user, grad_final_item)
            model.user_emb_base -= lr * grad_user_base
            model.item_emb_base -= lr * grad_item_base

        if verbose:
            print(f"epoch {epoch + 1}/{n_epochs}  avg_bpr_loss={epoch_loss / n_interactions:.4f}")

    model.forward()  # final propagation with the fully-trained base embeddings
    return model


if __name__ == "__main__":
    # Gradient check: verify backward() against numerical finite-difference
    # gradients on a tiny synthetic graph, before trusting it for training.
    rng = np.random.default_rng(0)
    n_users, n_items, n_factors, n_layers = 6, 5, 4, 2

    train_df = pd.DataFrame({
        "user_id": [0, 0, 1, 1, 2, 2, 3, 4, 5, 5],
        "item_id": [0, 1, 1, 2, 2, 3, 3, 4, 0, 4],
    })

    model = GCCF(n_users, n_items, n_factors, train_df, n_layers=n_layers, seed=0)

    def compute_loss():
        model.forward()
        # Simple, arbitrary differentiable scalar loss for the gradient check:
        # sum of squared dot products over a few (u,i) pairs.
        pairs = [(0, 0), (1, 2), (3, 3), (5, 4)]
        loss = 0.0
        for u, i in pairs:
            loss += 0.5 * (model.user_emb[u] @ model.item_emb[i]) ** 2
        return loss, pairs

    loss, pairs = compute_loss()
    print(f"Loss: {loss:.6f}")

    # Analytic gradient w.r.t. final embeddings for this toy loss.
    grad_final_user = np.zeros_like(model.user_emb)
    grad_final_item = np.zeros_like(model.item_emb)
    for u, i in pairs:
        s = model.user_emb[u] @ model.item_emb[i]
        grad_final_user[u] += s * model.item_emb[i]
        grad_final_item[i] += s * model.user_emb[u]

    grad_user_base, grad_item_base = model.backward(grad_final_user, grad_final_item)

    eps = 1e-5

    def numerical_grad(param, idx):
        grad = np.zeros_like(param[idx])
        for d in range(len(param[idx])):
            orig = param[idx][d]
            param[idx][d] = orig + eps
            loss_plus, _ = compute_loss()
            param[idx][d] = orig - eps
            loss_minus, _ = compute_loss()
            param[idx][d] = orig
            grad[d] = (loss_plus - loss_minus) / (2 * eps)
        return grad

    print("\nChecking grad_user_base for a few users:")
    for u in [0, 2, 5]:
        num_g = numerical_grad(model.user_emb_base, u)
        print(f"  user {u}: analytic={grad_user_base[u]}, numerical={num_g}, "
              f"max_diff={np.abs(grad_user_base[u]-num_g).max():.8f}")
        assert np.allclose(grad_user_base[u], num_g, atol=1e-4), f"user {u} grad mismatch!"

    print("\nChecking grad_item_base for a few items:")
    for i in [0, 2, 4]:
        num_g = numerical_grad(model.item_emb_base, i)
        print(f"  item {i}: analytic={grad_item_base[i]}, numerical={num_g}, "
              f"max_diff={np.abs(grad_item_base[i]-num_g).max():.8f}")
        assert np.allclose(grad_item_base[i], num_g, atol=1e-4), f"item {i} grad mismatch!"

    print("\nOK: GCCF backward pass matches numerical gradients exactly.")