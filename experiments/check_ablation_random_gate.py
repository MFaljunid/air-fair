"""
check_ablation_random_gate.py
---------------------------------
ABLATION: compares AIR-Fair's real quality-based gate against a RANDOMIZED
gate (same alpha_min/alpha_max distribution of correction strengths, but
shuffled across users -- severing the link between a user's TRUE need and
the correction they receive).

If the random gate performs similarly to the real gate, AIR-Fair's benefit
would be just from having ANY per-user variability, not from targeting the
right users -- undermining the whole design premise. If the real gate
clearly outperforms the random one, it demonstrates the SIGNAL (quality-
based targeting) itself is what matters, not just variance.

Run on the intersectional attribute (group_id, 8 groups), since Contribution
3 showed the clearest, most consistent pattern of any setting tested today.
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
TARGET_GROUP_LABEL = "F_50+"


def append_result_row(out_path: str, row: dict):
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(out_path)
    df_row.to_csv(out_path, mode="a", header=write_header, index=False)


def run(dataset_name: str = "ml-1m", positive_threshold: float = 3, seeds: list = None):
    seeds = seeds or [0, 1, 2]
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    raw_dir = os.path.join(project_root, "data", "raw", dataset_name)
    results_dir = os.path.join(project_root, "results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    out_path = get_versioned_path(os.path.join(results_dir, "ablation_random_gate.csv"))

    for seed in seeds:
        print(f"\n{'='*20} SEED={seed} {'='*20}")
        data = load_and_prepare(raw_dir=raw_dir, positive_threshold=positive_threshold, seed=seed)
        attr_result = prepare_all_attributes(data["user_attrs"])
        user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)

        target_rows = user_attrs_sorted[user_attrs_sorted["group_label"] == TARGET_GROUP_LABEL]
        target_group_id = target_rows["group_id"].iloc[0]
        is_target = (user_attrs_sorted["group_id"].values == target_group_id).astype(int)

        print(f"--- seed={seed}: Plain BPR ---")
        bpr_model = train_bpr(
            data["train"], data["n_users"], data["n_items"],
            n_factors=64, n_epochs=N_EPOCHS, seed=seed, verbose=False,
        )
        bpr_acc = evaluate_full_ranking(bpr_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST)
        bpr_fair = evaluate_fairness(bpr_model, data["train"], data["test"], is_target, data["n_users"], data["n_items"], k_list=K_LIST)

        print(f"--- seed={seed}: FairIR_BPR (strong) ---")
        fairir_model, _ = train_fairir(
            data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
            group_col="group_id", n_factors=64, n_epochs=N_EPOCHS,
            delta=STRONG_DELTA, alpha=STRONG_ALPHA, tau=0.7, n_negatives=5, seed=seed, verbose=False,
        )
        fairir_acc = evaluate_full_ranking(fairir_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST)
        fairir_fair = evaluate_fairness(fairir_model, data["train"], data["test"], is_target, data["n_users"], data["n_items"], k_list=K_LIST)

        print(f"--- seed={seed}: AIR-Fair (REAL gate) ---")
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

        print(f"--- seed={seed}: AIR-Fair (RANDOM gate, ablation) ---")
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
            bpr_hr = bpr_acc[f"HR@{k}"]
            row = {
                "seed": seed, "K": k,
                "BPR_HR": bpr_hr,
                "FairIR_HR_change_pct": (fairir_acc[f"HR@{k}"] - bpr_hr) / bpr_hr * 100,
                "RealGate_HR_change_pct": (real_acc[f"HR@{k}"] - bpr_hr) / bpr_hr * 100,
                "RandomGate_HR_change_pct": (random_acc[f"HR@{k}"] - bpr_hr) / bpr_hr * 100,
                "FairIR_DP": fairir_fair[f"DP@{k}"], "RealGate_DP": real_fair[f"DP@{k}"], "RandomGate_DP": random_fair[f"DP@{k}"],
            }
            append_result_row(out_path, row)
            print(f"  K={k}: HR change  FairIR={row['FairIR_HR_change_pct']:+.2f}%  "
                  f"RealGate={row['RealGate_HR_change_pct']:+.2f}%  RandomGate={row['RandomGate_HR_change_pct']:+.2f}%  |  "
                  f"DP  FairIR={row['FairIR_DP']:.4f}  RealGate={row['RealGate_DP']:.4f}  RandomGate={row['RandomGate_DP']:.4f}")

    print(f"\n{'='*70}\n=== Cross-seed summary: Real gate vs Random gate (ablation) ===")
    results_df = pd.read_csv(out_path)
    for k in K_LIST:
        sub = results_df[results_df["K"] == k]
        real_hr = sub["RealGate_HR_change_pct"].values
        random_hr = sub["RandomGate_HR_change_pct"].values
        real_vs_random = real_hr - random_hr
        print(f"\nK={k}: Real gate mean HR change={real_hr.mean():+.2f}%  "
              f"Random gate mean HR change={random_hr.mean():+.2f}%")
        print(f"  Real - Random per seed: {real_vs_random.round(2).tolist()}  mean={real_vs_random.mean():+.2f}pp  "
              f"({(real_vs_random>0).sum()}/{len(real_vs_random)} seeds favor REAL gate)")

    print(f"\nSaved to {out_path}")
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--positive-threshold", type=float, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    pos_thresh = None if args.positive_threshold == 0 else args.positive_threshold
    run(dataset_name=args.dataset, positive_threshold=pos_thresh, seeds=args.seeds)