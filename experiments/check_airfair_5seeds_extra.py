"""
check_airfair_wide_range.py
------------------------------
Tests whether a MUCH WIDER gate range (alpha_min=0.05 instead of 0.6) lets
AIR-Fair finally recover accuracy for well-served users -- our previous
test's range [0.6, 1.0] still applied substantial correction (>=60% noise)
to EVERY user, including the dominant/well-served group, which is likely
why HR never actually recovered versus fixed-strong FairIR. A properly wide
range should let well-served users keep near-zero correction (preserving
their accuracy close to plain BPR) while disadvantaged users still get up
to full correction -- the actual mechanism AIR-Fair's win-win claim depends on.
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
WIDE_ALPHA_MIN = 0.05
WIDE_ALPHA_MAX = 1.0
WIDE_DELTA_MIN_RATIO = 0.05
WIDE_DELTA_MAX_RATIO = 2.0


def append_result_row(out_path: str, row: dict):
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(out_path)
    df_row.to_csv(out_path, mode="a", header=write_header, index=False)


def run(dataset_name: str = "ml-1m", positive_threshold: float = 3, seeds: list = None):
    seeds = seeds or [2, 3, 4]
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    raw_dir = os.path.join(project_root, "data", "raw", dataset_name)
    results_dir = os.path.join(project_root, "results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    out_path = get_versioned_path(os.path.join(results_dir, "airfair_5seeds_extra_2_3_4.csv"))

    for seed in seeds:
        print(f"\n{'='*20} SEED={seed} {'='*20}")
        data = load_and_prepare(raw_dir=raw_dir, positive_threshold=positive_threshold, seed=seed)
        attr_result = prepare_all_attributes(data["user_attrs"])
        user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)
        user_group = user_attrs_sorted["gender_code"].values

        print(f"--- seed={seed}: Plain BPR ---")
        bpr_model = train_bpr(
            data["train"], data["n_users"], data["n_items"],
            n_factors=64, n_epochs=N_EPOCHS, seed=seed, verbose=False,
        )
        bpr_acc = evaluate_full_ranking(bpr_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST)
        bpr_fair = evaluate_fairness(bpr_model, data["train"], data["test"], user_group, data["n_users"], data["n_items"], k_list=K_LIST)

        print(f"--- seed={seed}: FairIR_BPR (strong, validated: alpha={STRONG_ALPHA}, delta={STRONG_DELTA}) ---")
        fairir_model, _ = train_fairir(
            data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
            group_col="gender_code", n_factors=64, n_epochs=N_EPOCHS,
            delta=STRONG_DELTA, alpha=STRONG_ALPHA, tau=0.7, n_negatives=5, seed=seed, verbose=False,
        )
        fairir_acc = evaluate_full_ranking(fairir_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST)
        fairir_fair = evaluate_fairness(fairir_model, data["train"], data["test"], user_group, data["n_users"], data["n_items"], k_list=K_LIST)

        print(f"--- seed={seed}: AIR-Fair (WIDE gate: alpha_min={WIDE_ALPHA_MIN}, alpha_max={WIDE_ALPHA_MAX}) ---")
        airfair_model, _, gate_info = train_airfair(
            data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
            group_col="gender_code", n_factors=64, n_epochs=N_EPOCHS,
            delta_base=STRONG_DELTA, alpha_base=STRONG_ALPHA,
            alpha_min=WIDE_ALPHA_MIN, alpha_max=WIDE_ALPHA_MAX,
            delta_min_ratio=WIDE_DELTA_MIN_RATIO, delta_max_ratio=WIDE_DELTA_MAX_RATIO,
            tau=0.7, n_negatives=5, seed=seed, verbose=False,
        )
        airfair_acc = evaluate_full_ranking(airfair_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST)
        airfair_fair = evaluate_fairness(airfair_model, data["train"], data["test"], user_group, data["n_users"], data["n_items"], k_list=K_LIST)

        for k in K_LIST:
            bpr_hr, fairir_hr, airfair_hr = bpr_acc[f"HR@{k}"], fairir_acc[f"HR@{k}"], airfair_acc[f"HR@{k}"]
            bpr_dp, fairir_dp, airfair_dp = bpr_fair[f"DP@{k}"], fairir_fair[f"DP@{k}"], airfair_fair[f"DP@{k}"]
            row = {
                "seed": seed, "K": k,
                "BPR_HR": bpr_hr, "FairIR_HR": fairir_hr, "AIRFair_HR": airfair_hr,
                "FairIR_HR_change_pct": (fairir_hr - bpr_hr) / bpr_hr * 100,
                "AIRFair_HR_change_pct": (airfair_hr - bpr_hr) / bpr_hr * 100,
                "BPR_DP": bpr_dp, "FairIR_DP": fairir_dp, "AIRFair_DP": airfair_dp,
                "FairIR_DP_reduction_pct": (bpr_dp - fairir_dp) / bpr_dp * 100,
                "AIRFair_DP_reduction_pct": (bpr_dp - airfair_dp) / bpr_dp * 100,
            }
            append_result_row(out_path, row)
            print(f"  K={k}: HR change  FairIR={row['FairIR_HR_change_pct']:+.2f}%  AIR-Fair={row['AIRFair_HR_change_pct']:+.2f}%  |  "
                  f"DP reduction  FairIR={row['FairIR_DP_reduction_pct']:+.2f}%  AIR-Fair={row['AIRFair_DP_reduction_pct']:+.2f}%")

    print(f"\n{'='*70}\n=== Cross-seed summary (wide gate range) ===")
    results_df = pd.read_csv(out_path)
    for k in K_LIST:
        sub = results_df[results_df["K"] == k]
        print(f"\nK={k}:")
        print(f"  FairIR: mean HR change={sub['FairIR_HR_change_pct'].mean():+.2f}%  mean DP reduction={sub['FairIR_DP_reduction_pct'].mean():+.2f}%")
        print(f"  AIR-Fair (wide): mean HR change={sub['AIRFair_HR_change_pct'].mean():+.2f}%  mean DP reduction={sub['AIRFair_DP_reduction_pct'].mean():+.2f}%")
        hr_recovered = sub['AIRFair_HR_change_pct'].mean() - sub['FairIR_HR_change_pct'].mean()
        dp_diff = sub['AIRFair_DP_reduction_pct'].mean() - sub['FairIR_DP_reduction_pct'].mean()
        win_win = (hr_recovered > 0) and (dp_diff > 0)
        print(f"  -> HR recovered={hr_recovered:+.2f}pp, DP gain diff={dp_diff:+.2f}pp  {'*** WIN-WIN ***' if win_win else ''}")

    print(f"\nSaved to {out_path}")
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--positive-threshold", type=float, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4])
    args = parser.parse_args()
    pos_thresh = None if args.positive_threshold == 0 else args.positive_threshold
    run(dataset_name=args.dataset, positive_threshold=pos_thresh, seeds=args.seeds)
