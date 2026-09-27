"""
check_gccf_degradation.py
----------------------------
Tests whether FairIR's own documented GCCF degradation (Section 4.4: message
passing dilutes the fairness signal) reproduces in OUR reimplementation, on
real ML-1M data, BEFORE building any fix for it.

Compares 4 models under the same protocol: BPR, GCCF (plain), FairIR_BPR,
FairIR_GCCF. FairIR's own claim (their Tables 3-5) is that the FAIRNESS GAIN
(DP/EO reduction vs the plain backbone) is smaller for GCCF than for BPR.
We check the exact same pattern here.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from preprocessing.load_data import load_and_prepare
from preprocessing.attributes import prepare_all_attributes
from models.mf import train_bpr
from models.gccf import train_gccf_bpr
from training.train_fairir import train_fairir
from training.train_fairir_gccf import train_fairir_gccf
from eval.metrics import evaluate_full_ranking, evaluate_fairness
from versioning import get_versioned_path

K_LIST = [10, 20, 30, 40]


def run(dataset_name: str = "ml-1m", positive_threshold: float = 3, seed: int = 0):
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    raw_dir = os.path.join(project_root, "data", "raw", dataset_name)
    results_dir = os.path.join(project_root, "results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"=== Dataset: {dataset_name} ===")
    data = load_and_prepare(raw_dir=raw_dir, positive_threshold=positive_threshold, seed=seed)
    print(f"n_users={data['n_users']}, n_items={data['n_items']}, train={len(data['train'])}")

    attr_result = prepare_all_attributes(data["user_attrs"])
    user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)
    user_group = user_attrs_sorted["gender_code"].values

    results = {}

    print("\n=== [1/4] Plain BPR ===")
    m = train_bpr(data["train"], data["n_users"], data["n_items"], n_factors=64, n_epochs=15, seed=seed)
    results["BPR"] = (
        evaluate_full_ranking(m, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST),
        evaluate_fairness(m, data["train"], data["test"], user_group, data["n_users"], data["n_items"], k_list=K_LIST),
    )

    print("\n=== [2/4] Plain GCCF ===")
    m = train_gccf_bpr(data["train"], data["n_users"], data["n_items"], n_factors=64, n_layers=2, n_epochs=15, seed=seed)
    results["GCCF"] = (
        evaluate_full_ranking(m, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST),
        evaluate_fairness(m, data["train"], data["test"], user_group, data["n_users"], data["n_items"], k_list=K_LIST),
    )

    print("\n=== [3/4] FairIR_BPR ===")
    m, _ = train_fairir(
        data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
        group_col="gender_code", n_factors=64, n_epochs=15,
        delta=0.1, alpha=0.5, tau=0.7, n_negatives=5, seed=seed,
    )
    results["FairIR_BPR"] = (
        evaluate_full_ranking(m, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST),
        evaluate_fairness(m, data["train"], data["test"], user_group, data["n_users"], data["n_items"], k_list=K_LIST),
    )

    print("\n=== [4/4] FairIR_GCCF ===")
    m, _ = train_fairir_gccf(
        data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
        group_col="gender_code", n_factors=64, n_layers=2, n_epochs=15,
        delta=0.1, alpha=0.5, tau=0.7, n_negatives=5, seed=seed,
    )
    results["FairIR_GCCF"] = (
        evaluate_full_ranking(m, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST),
        evaluate_fairness(m, data["train"], data["test"], user_group, data["n_users"], data["n_items"], k_list=K_LIST),
    )

    print("\n" + "=" * 70)
    print("=== FINAL COMPARISON (all K) ===")
    for k in K_LIST:
        print(f"\n--- K={k} ---")
        print(f"{'Model':<14} {'HR':>10} {'NDCG':>10} {'DP':>10} {'EO':>10}")
        for model_name, (acc, fair) in results.items():
            print(f"{model_name:<14} {acc[f'HR@{k}']:>10.4f} {acc[f'NDCG@{k}']:>10.4f} "
                  f"{fair[f'DP@{k}']:>10.4f} {fair[f'EO@{k}']:>10.4f}")

    print("\n--- DP reduction vs backbone (FairIR's core claim: BPR gain > GCCF gain) ---")
    for k in K_LIST:
        bpr_dp = results["BPR"][1][f"DP@{k}"]
        fairir_bpr_dp = results["FairIR_BPR"][1][f"DP@{k}"]
        gccf_dp = results["GCCF"][1][f"DP@{k}"]
        fairir_gccf_dp = results["FairIR_GCCF"][1][f"DP@{k}"]
        bpr_gain = (bpr_dp - fairir_bpr_dp) / bpr_dp * 100
        gccf_gain = (gccf_dp - fairir_gccf_dp) / gccf_dp * 100
        print(f"K={k}: BPR DP reduction={bpr_gain:+.2f}%, GCCF DP reduction={gccf_gain:+.2f}% "
              f"{'(degradation confirmed: GCCF gain < BPR gain)' if gccf_gain < bpr_gain else '(NOT confirmed here)'}")

    rows = []
    for model_name, (acc, fair) in results.items():
        for k in K_LIST:
            rows.append({
                "dataset": dataset_name, "model": model_name, "K": k,
                "HR": acc[f"HR@{k}"], "NDCG": acc[f"NDCG@{k}"],
                "DP": fair[f"DP@{k}"], "EO": fair[f"EO@{k}"],
            })
    results_df = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "gccf_degradation_check.csv")
    out_path = get_versioned_path(out_path)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--positive-threshold", type=float, default=3)
    args = parser.parse_args()
    pos_thresh = None if args.positive_threshold == 0 else args.positive_threshold
    run(dataset_name=args.dataset, positive_threshold=pos_thresh)