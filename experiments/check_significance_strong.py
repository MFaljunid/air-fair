"""
check_significance_strong.py
-------------------------------
Same multi-seed significance check as check_significance.py, but testing
whether a STRONGER intervention (alpha=0.8, delta=0.5 instead of 0.5/0.1)
trained for MORE epochs (30 instead of 15) produces a larger, more
consistently-detectable DP effect -- the two most likely reasons today's
effect was too weak/noisy to reliably detect across seeds.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from preprocessing.load_data import load_and_prepare
from preprocessing.attributes import prepare_all_attributes
from models.mf import train_bpr
from training.train_fairir import train_fairir
from eval.metrics import evaluate_full_ranking, evaluate_fairness
from versioning import get_versioned_path

K_LIST = [10, 20, 30, 40]
SEEDS = [0, 1, 2]
N_EPOCHS = 30
ALPHA = 0.8
DELTA = 0.5


def append_result_row(out_path: str, row: dict):
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(out_path)
    df_row.to_csv(out_path, mode="a", header=write_header, index=False)


def run(dataset_name: str = "ml-1m", positive_threshold: float = 3, seeds: list = None):
    seeds = seeds or SEEDS
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    raw_dir = os.path.join(project_root, "data", "raw", dataset_name)
    results_dir = os.path.join(project_root, "results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    out_path = get_versioned_path(os.path.join(results_dir, "significance_check_strong.csv"))

    for seed in seeds:
        print(f"\n{'='*20} SEED={seed} (alpha={ALPHA}, delta={DELTA}, epochs={N_EPOCHS}) {'='*20}")
        data = load_and_prepare(raw_dir=raw_dir, positive_threshold=positive_threshold, seed=seed)
        attr_result = prepare_all_attributes(data["user_attrs"])
        user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)
        user_group = user_attrs_sorted["gender_code"].values

        print(f"--- seed={seed}: Plain BPR ({N_EPOCHS} epochs) ---")
        bpr_model = train_bpr(
            data["train"], data["n_users"], data["n_items"],
            n_factors=64, n_epochs=N_EPOCHS, seed=seed, verbose=False,
        )
        bpr_acc = evaluate_full_ranking(bpr_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST)
        bpr_fair = evaluate_fairness(bpr_model, data["train"], data["test"], user_group, data["n_users"], data["n_items"], k_list=K_LIST)

        print(f"--- seed={seed}: FairIR_BPR (strong: alpha={ALPHA}, delta={DELTA}) ---")
        fairir_model, _ = train_fairir(
            data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
            group_col="gender_code", n_factors=64, n_epochs=N_EPOCHS,
            delta=DELTA, alpha=ALPHA, tau=0.7, n_negatives=5, seed=seed, verbose=False,
        )
        fairir_acc = evaluate_full_ranking(fairir_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST)
        fairir_fair = evaluate_fairness(fairir_model, data["train"], data["test"], user_group, data["n_users"], data["n_items"], k_list=K_LIST)

        for k in K_LIST:
            bpr_dp, fairir_dp = bpr_fair[f"DP@{k}"], fairir_fair[f"DP@{k}"]
            dp_reduction_pct = (bpr_dp - fairir_dp) / bpr_dp * 100
            bpr_hr, fairir_hr = bpr_acc[f"HR@{k}"], fairir_acc[f"HR@{k}"]
            row = {
                "seed": seed, "K": k,
                "BPR_HR": bpr_hr, "FairIR_HR": fairir_hr,
                "HR_change_pct": (fairir_hr - bpr_hr) / bpr_hr * 100,
                "BPR_DP": bpr_dp, "FairIR_DP": fairir_dp, "DP_reduction_pct": dp_reduction_pct,
                "BPR_EO": bpr_fair[f"EO@{k}"], "FairIR_EO": fairir_fair[f"EO@{k}"],
            }
            append_result_row(out_path, row)
            print(f"  K={k}: DP {bpr_dp:.4f}->{fairir_dp:.4f} ({dp_reduction_pct:+.2f}%)  "
                  f"HR change: {row['HR_change_pct']:+.2f}%")

    print(f"\n{'='*50}\n=== Cross-seed summary (strong intervention) ===")
    results_df = pd.read_csv(out_path)
    for k in K_LIST:
        sub = results_df[results_df["K"] == k]
        reductions = sub["DP_reduction_pct"].values
        hr_changes = sub["HR_change_pct"].values
        consistent = (reductions > 0).all() or (reductions < 0).all()
        print(f"K={k}: DP reduction per seed = {reductions.round(2).tolist()}%  "
              f"mean={reductions.mean():+.2f}%  std={reductions.std():.2f}%  "
              f"{'CONSISTENT' if consistent else 'INCONSISTENT'}  |  "
              f"mean HR change={hr_changes.mean():+.2f}%")

    print(f"\nSaved to {out_path}")
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--positive-threshold", type=float, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    pos_thresh = None if args.positive_threshold == 0 else args.positive_threshold
    run(dataset_name=args.dataset, positive_threshold=pos_thresh, seeds=args.seeds)