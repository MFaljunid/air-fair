"""
run_fairir_baseline.py
------------------------
Full end-to-end run of our FairIR reimplementation on a given dataset:
load (with the FairIR-matching positive_threshold=3 filter for MovieLens) ->
train FairIR (BPR + cross-group noise injection + contrastive distillation)
-> evaluate HR@K/NDCG@K/DP@K/EO@K under our LOO + full-ranking protocol.

Also trains a PLAIN BPR baseline (no fairness mechanism) on the exact same
data/split for a direct before/after comparison within our own pipeline.

Results are saved PER DATASET under results/<dataset_name>/ (with a
figures/ subfolder alongside it), so running multiple datasets never
overwrites or mixes results together -- matches the same convention used
in the earlier CD-AFMS project.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from versioning import get_versioned_path

import pandas as pd

from preprocessing.load_data import load_and_prepare
from preprocessing.attributes import prepare_all_attributes
from models.mf import train_bpr
from training.train_fairir import train_fairir
from eval.metrics import evaluate_full_ranking, evaluate_fairness

K_LIST = [10, 20, 30, 40]


def run(dataset_name: str, positive_threshold: float = None, seed: int = 0):
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    raw_dir = os.path.join(project_root, "data", "raw", dataset_name)
    results_dir = os.path.join(project_root, "results", dataset_name)
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"=== Dataset: {dataset_name} ===")
    print(f"=== Loading (positive_threshold={positive_threshold}) ===")
    data = load_and_prepare(raw_dir=raw_dir, positive_threshold=positive_threshold, seed=seed)
    print(f"n_users={data['n_users']}, n_items={data['n_items']}, train={len(data['train'])}")

    attr_result = prepare_all_attributes(data["user_attrs"])
    user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)
    user_group = user_attrs_sorted["gender_code"].values

    print("\n=== Training PLAIN BPR (no fairness mechanism) ===")
    bpr_model = train_bpr(
        data["train"], data["n_users"], data["n_items"],
        n_factors=64, n_epochs=15, seed=seed,
    )
    bpr_acc = evaluate_full_ranking(
        bpr_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST
    )
    bpr_fair = evaluate_fairness(
        bpr_model, data["train"], data["test"], user_group,
        data["n_users"], data["n_items"], k_list=K_LIST,
    )

    print("\n=== Training FairIR (BPR + cross-group noise + contrastive) ===")
    fairir_model, noise_module = train_fairir(
        data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
        group_col="gender_code",
        n_factors=64, n_epochs=15,
        delta=0.1, alpha=0.5, tau=0.7, n_negatives=5,
        seed=seed,
    )
    fairir_acc = evaluate_full_ranking(
        fairir_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST
    )
    fairir_fair = evaluate_fairness(
        fairir_model, data["train"], data["test"], user_group,
        data["n_users"], data["n_items"], k_list=K_LIST,
    )

    print("\n" + "=" * 60)
    print("=== FINAL COMPARISON (K=10) ===")
    print(f"{'Metric':<10} {'Plain BPR':>12} {'FairIR (ours)':>15}")
    for metric in ["HR@10", "NDCG@10"]:
        print(f"{metric:<10} {bpr_acc[metric]:>12.4f} {fairir_acc[metric]:>15.4f}")
    for metric in ["DP@10", "EO@10"]:
        print(f"{metric:<10} {bpr_fair[metric]:>12.4f} {fairir_fair[metric]:>15.4f}")

    # Build one tidy results table (long format: one row per model x metric x K)
    rows = []
    for model_name, acc, fair in [("BPR", bpr_acc, bpr_fair), ("FairIR", fairir_acc, fairir_fair)]:
        for k in K_LIST:
            rows.append({
                "dataset": dataset_name, "model": model_name, "K": k,
                "HR": acc[f"HR@{k}"], "NDCG": acc[f"NDCG@{k}"],
                "DP": fair[f"DP@{k}"], "EO": fair[f"EO@{k}"],
            })
    results_df = pd.DataFrame(rows)

    out_path = os.path.join(results_dir, "fairir_baseline_results.csv")
    out_path = get_versioned_path(out_path)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(f"(figures/ folder ready at {figures_dir} for later plots)")

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m",
                         help="Dataset folder name under data/raw/")
    parser.add_argument("--positive-threshold", type=float, default=3,
                         help="Rating threshold for positive feedback (MovieLens-style); "
                              "pass e.g. --positive-threshold 0 to disable for datasets "
                              "like LastFM whose 'rating' isn't a 1-5 scale.")
    args = parser.parse_args()

    pos_thresh = None if args.positive_threshold == 0 else args.positive_threshold
    run(dataset_name=args.dataset, positive_threshold=pos_thresh)