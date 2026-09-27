"""
diagnostic_lastfm_figures_data.py
-------------------------------------
Same purpose as diagnostic_ml1m_figures_data.py, but for LastFM: a single-
seed diagnostic run (illustrative, not a statistical claim -- multi-seed
tables remain the source of truth) to capture:
  1. The full per-user alpha_u array from AIR-Fair's gate on LastFM.
  2. EO@K alongside DP@K for BPR/FairIR/AIR-Fair on LastFM.

Saves alpha_u to a .npy file and a full DP/EO CSV, mirroring the ML-1M
diagnostic exactly so the two can be plotted side by side.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from preprocessing.load_data import load_and_prepare
from preprocessing.attributes import prepare_all_attributes
from models.mf import train_bpr
from training.train_fairir import train_fairir
from training.train_airfair import train_airfair
from eval.metrics import evaluate_full_ranking, evaluate_fairness

K_LIST = [10, 20, 30, 40]
N_EPOCHS = 30
STRONG_ALPHA = 0.8
STRONG_DELTA = 0.5
GATE_ALPHA_MIN = 0.05
GATE_ALPHA_MAX = 1.0
SEED = 0

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
raw_dir = os.path.join(project_root, "data", "raw", "lastfm")
out_dir = os.path.join(project_root, "results", "lastfm", "diagnostics")
os.makedirs(out_dir, exist_ok=True)

print("=== Loading LastFM (final recipe) ===")
data = load_and_prepare(
    raw_dir=raw_dir, positive_threshold=None,
    min_user_interactions=40, min_item_interactions=10,
    subsample_users=15000, seed=SEED,
)
attr_result = prepare_all_attributes(data["user_attrs"])
user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)
gender = user_attrs_sorted["gender_code"].values

print("\n=== [1/3] Plain BPR ===")
bpr_model = train_bpr(data["train"], data["n_users"], data["n_items"],
                       n_factors=64, n_epochs=N_EPOCHS, seed=SEED, verbose=False)
bpr_fair = evaluate_fairness(bpr_model, data["train"], data["test"], gender, data["n_users"], data["n_items"], k_list=K_LIST)

print("=== [2/3] FairIR (strong) ===")
fairir_model, _ = train_fairir(data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
                                group_col="gender_code", n_factors=64, n_epochs=N_EPOCHS,
                                delta=STRONG_DELTA, alpha=STRONG_ALPHA, tau=0.7, n_negatives=5, seed=SEED, verbose=False)
fairir_fair = evaluate_fairness(fairir_model, data["train"], data["test"], gender, data["n_users"], data["n_items"], k_list=K_LIST)

print("=== [3/3] AIR-Fair (wide gate) ===")
airfair_model, _, gate_info = train_airfair(data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
                                             group_col="gender_code", n_factors=64, n_epochs=N_EPOCHS,
                                             delta_base=STRONG_DELTA, alpha_base=STRONG_ALPHA,
                                             alpha_min=GATE_ALPHA_MIN, alpha_max=GATE_ALPHA_MAX,
                                             tau=0.7, n_negatives=5, seed=SEED, verbose=False)
airfair_fair = evaluate_fairness(airfair_model, data["train"], data["test"], gender, data["n_users"], data["n_items"], k_list=K_LIST)

alpha_u = gate_info["alpha_u"]
np.save(os.path.join(out_dir, "alpha_u_lastfm_seed0.npy"), alpha_u)
dominant_group = gate_info["dominant_group"]
print(f"\nDominant group: {dominant_group}")
print(f"Group 0 (n={np.sum(gender==0)}) mean alpha_u: {alpha_u[gender==0].mean():.4f}  std: {alpha_u[gender==0].std():.4f}")
print(f"Group 1 (n={np.sum(gender==1)}) mean alpha_u: {alpha_u[gender==1].mean():.4f}  std: {alpha_u[gender==1].std():.4f}")
print(f"Overall alpha_u: min={alpha_u.min():.4f}  max={alpha_u.max():.4f}  saved to alpha_u_lastfm_seed0.npy")

print("\n=== DP + EO comparison (BPR vs FairIR vs AIR-Fair, LastFM, seed=0) ===")
rows = []
for k in K_LIST:
    row = {
        "K": k,
        "BPR_DP": bpr_fair[f"DP@{k}"], "BPR_EO": bpr_fair[f"EO@{k}"],
        "FairIR_DP": fairir_fair[f"DP@{k}"], "FairIR_EO": fairir_fair[f"EO@{k}"],
        "AIRFair_DP": airfair_fair[f"DP@{k}"], "AIRFair_EO": airfair_fair[f"EO@{k}"],
    }
    rows.append(row)
    print(f"K={k}: BPR(DP={row['BPR_DP']:.4f}, EO={row['BPR_EO']:.4f})  "
          f"FairIR(DP={row['FairIR_DP']:.4f}, EO={row['FairIR_EO']:.4f})  "
          f"AIR-Fair(DP={row['AIRFair_DP']:.4f}, EO={row['AIRFair_EO']:.4f})")

pd.DataFrame(rows).to_csv(os.path.join(out_dir, "dp_eo_comparison_lastfm_seed0.csv"), index=False)
print(f"\nSaved to {out_dir}/dp_eo_comparison_lastfm_seed0.csv")