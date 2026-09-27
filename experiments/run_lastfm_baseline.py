"""
run_lastfm_baseline.py
-------------------------
First real experiment on LastFM: Plain BPR vs FairIR_BPR (strong, validated
settings: alpha=0.8, delta=0.5, 30 epochs) using the final agreed
preprocessing recipe (min_user=40, min_item=10 core filter, then subsampled
to 15,000 users for compute feasibility). Same LOO + full-ranking evaluation
protocol used throughout the ML-1M experiments.

Results saved to results/lastfm/ (same per-dataset convention as ML-1M).
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
N_EPOCHS = 30
STRONG_ALPHA = 0.8
STRONG_DELTA = 0.5


def run(dataset_name: str = "lastfm", seed: int = 0):
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    raw_dir = os.path.join(project_root, "data", "raw", dataset_name)
    results_dir = os.path.join(project_root, "results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"=== Dataset: {dataset_name} (final recipe: min_user=40, min_item=10, subsample=15000) ===")
    data = load_and_prepare(
        raw_dir=raw_dir, positive_threshold=None,
        min_user_interactions=40, min_item_interactions=10,
        subsample_users=15000, seed=seed,
    )
    print(f"n_users={data['n_users']}, n_items={data['n_items']}, train={len(data['train'])}")

    attr_result = prepare_all_attributes(data["user_attrs"])
    user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)
    user_group = user_attrs_sorted["gender_code"].values

    print("\n=== [1/2] Plain BPR ===")
    bpr_model = train_bpr(
        data["train"], data["n_users"], data["n_items"],
        n_factors=64, n_epochs=N_EPOCHS, seed=seed,
    )
    bpr_acc = evaluate_full_ranking(bpr_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST)
    bpr_fair = evaluate_fairness(bpr_model, data["train"], data["test"], user_group, data["n_users"], data["n_items"], k_list=K_LIST)

    print("\n=== [2/2] FairIR_BPR (strong: alpha=0.8, delta=0.5) ===")
    fairir_model, _ = train_fairir(
        data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
        group_col="gender_code", n_factors=64, n_epochs=N_EPOCHS,
        delta=STRONG_DELTA, alpha=STRONG_ALPHA, tau=0.7, n_negatives=5, seed=seed,
    )
    fairir_acc = evaluate_full_ranking(fairir_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST)
    fairir_fair = evaluate_fairness(fairir_model, data["train"], data["test"], user_group, data["n_users"], data["n_items"], k_list=K_LIST)

    print("\n" + "=" * 60)
    print("=== FINAL COMPARISON (all K) ===")
    for k in K_LIST:
        print(f"\n--- K={k} ---")
        print(f"{'Model':<10} {'HR':>10} {'NDCG':>10} {'DP':>10} {'EO':>10}")
        print(f"{'BPR':<10} {bpr_acc[f'HR@{k}']:>10.4f} {bpr_acc[f'NDCG@{k}']:>10.4f} "
              f"{bpr_fair[f'DP@{k}']:>10.4f} {bpr_fair[f'EO@{k}']:>10.4f}")
        print(f"{'FairIR':<10} {fairir_acc[f'HR@{k}']:>10.4f} {fairir_acc[f'NDCG@{k}']:>10.4f} "
              f"{fairir_fair[f'DP@{k}']:>10.4f} {fairir_fair[f'EO@{k}']:>10.4f}")
        bpr_dp = bpr_fair[f"DP@{k}"]
        dp_reduction = (bpr_dp - fairir_fair[f"DP@{k}"]) / bpr_dp * 100
        print(f"  DP reduction: {dp_reduction:+.2f}%")

    rows = []
    for model_name, acc, fair in [("BPR", bpr_acc, bpr_fair), ("FairIR", fairir_acc, fairir_fair)]:
        for k in K_LIST:
            rows.append({
                "dataset": dataset_name, "model": model_name, "seed": seed, "K": k,
                "HR": acc[f"HR@{k}"], "NDCG": acc[f"NDCG@{k}"],
                "DP": fair[f"DP@{k}"], "EO": fair[f"EO@{k}"],
            })
    results_df = pd.DataFrame(rows)
    out_path = get_versioned_path(os.path.join(results_dir, "lastfm_baseline_results.csv"))
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(seed=args.seed)