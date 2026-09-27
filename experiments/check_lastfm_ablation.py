"""
check_lastfm_ablation.py
----------------------------
CONTRIBUTION 4 on LastFM: same test as check_ablation_random_gate.py on
ML-1M -- does AIR-Fair's real (quality-based) gate outperform a randomized
gate of identical statistical range, on LastFM's final recipe (min_user=40,
min_item=10 core filter, invalid age/gender excluded, subsampled to 15,000
users)? Target group: M_50+ (179 users), same as the intersectional
contribution, for consistency.

Each seed's result is saved immediately.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from preprocessing.load_data import load_and_prepare
from preprocessing.attributes import prepare_all_attributes
from models.mf import train_bpr
from training.train_fairir import train_fairir
from training.train_airfair import train_airfair
from eval.metrics import evaluate_full_ranking, evaluate_fairness
from versioning import get_versioned_path

K_LIST = [10, 20, 30, 40]
N_EPOCHS = 30
STRONG_ALPHA = 0.8
STRONG_DELTA = 0.5
GATE_ALPHA_MIN = 0.05
GATE_ALPHA_MAX = 1.0
GATE_DELTA_MIN_RATIO = 0.05
GATE_DELTA_MAX_RATIO = 2.0
TARGET_GROUP_LABEL = "M_50+"


def append_result_row(out_path: str, row: dict):
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(out_path)
    df_row.to_csv(out_path, mode="a", header=write_header, index=False)


def run(dataset_name: str = "lastfm", seeds: list = None):
    seeds = seeds or [0, 1, 2]
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    raw_dir = os.path.join(project_root, "data", "raw", dataset_name)
    results_dir = os.path.join(project_root, "results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    out_path = get_versioned_path(os.path.join(results_dir, "ablation_random_gate.csv"))

    for seed in seeds:
        print(f"\n{'='*20} SEED={seed} (ablation: real vs random gate, target={TARGET_GROUP_LABEL}) {'='*20}")
        data = load_and_prepare(
            raw_dir=raw_dir, positive_threshold=None,
            min_user_interactions=40, min_item_interactions=10,
            subsample_users=15000, seed=seed,
        )
        attr_result = prepare_all_attributes(data["user_attrs"])
        user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)

        target_rows = user_attrs_sorted[user_attrs_sorted["group_label"] == TARGET_GROUP_LABEL]
        if len(target_rows) == 0:
            raise ValueError(f"Target group label {TARGET_GROUP_LABEL!r} not found in this seed's data.")
        target_group_id = target_rows["group_id"].iloc[0]
        is_target = (user_attrs_sorted["group_id"].values == target_group_id).astype(int)
        print(f"Target group '{TARGET_GROUP_LABEL}' (group_id={target_group_id}): n={is_target.sum()} users")

        print(f"--- seed={seed}: Plain BPR ---")
        bpr_model = train_bpr(
            data["train"], data["n_users"], data["n_items"],
            n_factors=64, n_epochs=N_EPOCHS, seed=seed, verbose=False,
        )
        bpr_acc = evaluate_full_ranking(bpr_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST)

        print(f"--- seed={seed}: FairIR_BPR (strong, group_col=group_id) ---")
        fairir_model, _ = train_fairir(
            data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
            group_col="group_id", n_factors=64, n_epochs=N_EPOCHS,
            delta=STRONG_DELTA, alpha=STRONG_ALPHA, tau=0.7, n_negatives=5, seed=seed, verbose=False,
        )
        fairir_acc = evaluate_full_ranking(fairir_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST)

        print(f"--- seed={seed}: AIR-Fair REAL gate (group_col=group_id) ---")
        real_model, _, _ = train_airfair(
            data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
            group_col="group_id", n_factors=64, n_epochs=N_EPOCHS,
            delta_base=STRONG_DELTA, alpha_base=STRONG_ALPHA,
            alpha_min=GATE_ALPHA_MIN, alpha_max=GATE_ALPHA_MAX,
            delta_min_ratio=GATE_DELTA_MIN_RATIO, delta_max_ratio=GATE_DELTA_MAX_RATIO,
            tau=0.7, n_negatives=5, seed=seed, verbose=False, randomize_gate=False,
        )
        real_acc = evaluate_full_ranking(real_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST)
        real_fair = evaluate_fairness(real_model, data["train"], data["test"], is_target, data["n_users"], data["n_items"], k_list=K_LIST)

        print(f"--- seed={seed}: AIR-Fair RANDOM gate (group_col=group_id) ---")
        random_model, _, _ = train_airfair(
            data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
            group_col="group_id", n_factors=64, n_epochs=N_EPOCHS,
            delta_base=STRONG_DELTA, alpha_base=STRONG_ALPHA,
            alpha_min=GATE_ALPHA_MIN, alpha_max=GATE_ALPHA_MAX,
            delta_min_ratio=GATE_DELTA_MIN_RATIO, delta_max_ratio=GATE_DELTA_MAX_RATIO,
            tau=0.7, n_negatives=5, seed=seed, verbose=False, randomize_gate=True,
        )
        random_acc = evaluate_full_ranking(random_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST)
        random_fair = evaluate_fairness(random_model, data["train"], data["test"], is_target, data["n_users"], data["n_items"], k_list=K_LIST)

        for k in K_LIST:
            bpr_hr, fairir_hr = bpr_acc[f"HR@{k}"], fairir_acc[f"HR@{k}"]
            real_hr, random_hr = real_acc[f"HR@{k}"], random_acc[f"HR@{k}"]
            real_dp, random_dp = real_fair[f"DP@{k}"], random_fair[f"DP@{k}"]
            row = {
                "seed": seed, "K": k,
                "BPR_HR": bpr_hr, "FairIR_HR": fairir_hr,
                "RealGate_HR": real_hr, "RandomGate_HR": random_hr,
                "FairIR_HR_change_pct": (fairir_hr - bpr_hr) / bpr_hr * 100,
                "RealGate_HR_change_pct": (real_hr - bpr_hr) / bpr_hr * 100,
                "RandomGate_HR_change_pct": (random_hr - bpr_hr) / bpr_hr * 100,
                "RealGate_DP": real_dp, "RandomGate_DP": random_dp,
            }
            append_result_row(out_path, row)
            print(f"  K={k}: FairIR={row['FairIR_HR_change_pct']:+.2f}%  RealGate={row['RealGate_HR_change_pct']:+.2f}%  "
                  f"RandomGate={row['RandomGate_HR_change_pct']:+.2f}%  (Real-Random={row['RealGate_HR_change_pct']-row['RandomGate_HR_change_pct']:+.2f}pp)")

    print(f"\n{'='*70}\n=== Cross-seed summary (LastFM ablation, target={TARGET_GROUP_LABEL}) ===")
    results_df = pd.read_csv(out_path)
    for k in K_LIST:
        sub = results_df[results_df["K"] == k]
        diff = sub["RealGate_HR_change_pct"].values - sub["RandomGate_HR_change_pct"].values
        print(f"K={k}: Real-Random per seed = {diff.round(2).tolist()}pp  mean={diff.mean():+.2f}pp  "
              f"std={diff.std():.2f}pp  ({(diff>0).sum()}/{len(diff)} seeds favoring real)")

    print(f"\nSaved to {out_path}")
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    run(seeds=args.seeds)