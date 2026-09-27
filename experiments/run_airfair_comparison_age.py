"""
run_airfair_comparison_age.py
--------------------------------
Same three-way comparison as run_airfair_comparison.py, but targeting AGE
instead of gender -- motivated by check_gate_gap_age.py showing "50+" has
the largest quality gap we've found (relative_gap=0.1355 vs gender's 0.0841),
consistent with the earlier CD-AFMS finding that "50+" was the most
statistically distinct age group on ML-1M.

Training uses the FULL 4-valued age_group_code (cross-group noise pools all
3 other age groups; the instance gate auto-detects the dominant group) --
our code already supports this with zero changes, since group_col was never
hardcoded to binary. DP/EO evaluation, however, IS binary-only in
eval/metrics.py, so fairness is measured as "50+ vs everyone else"
(is_senior), the group where the gap is largest.
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
from training.train_airfair import train_airfair
from eval.metrics import evaluate_full_ranking, evaluate_fairness

K_LIST = [10, 20, 30, 40]
SENIOR_AGE_GROUP_CODE = 3  # "50+" in attributes.py's AGE_LABELS ordering


def run(dataset_name: str = "ml-1m", positive_threshold: float = 3, invert_gate: bool = False, seed: int = 0):
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    raw_dir = os.path.join(project_root, "data", "raw", dataset_name)
    results_dir = os.path.join(project_root, "results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"=== Dataset: {dataset_name} (target attribute: age, 50+ vs rest) ===")
    data = load_and_prepare(raw_dir=raw_dir, positive_threshold=positive_threshold, seed=seed)
    print(f"n_users={data['n_users']}, n_items={data['n_items']}, train={len(data['train'])}")

    attr_result = prepare_all_attributes(data["user_attrs"])
    user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)

    # Full 4-valued age group for TRAINING (cross-group noise + gate).
    age_group_full = user_attrs_sorted["age_group_code"].values
    # Binary "is_senior" (50+ vs rest) for DP/EO EVALUATION only.
    is_senior = (age_group_full == SENIOR_AGE_GROUP_CODE).astype(int)
    print(f"n seniors (50+): {is_senior.sum()}, n non-seniors: {(1 - is_senior).sum()}")

    results = {}

    print("\n=== [1/3] Plain BPR ===")
    bpr_model = train_bpr(
        data["train"], data["n_users"], data["n_items"],
        n_factors=64, n_epochs=15, seed=seed,
    )
    results["BPR"] = (
        evaluate_full_ranking(bpr_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST),
        evaluate_fairness(bpr_model, data["train"], data["test"], is_senior, data["n_users"], data["n_items"], k_list=K_LIST),
    )

    print("\n=== [2/3] FairIR (fixed alpha=0.5, tau=0.7, group_col=age_group_code) ===")
    fairir_model, _ = train_fairir(
        data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
        group_col="age_group_code", n_factors=64, n_epochs=15,
        delta=0.1, alpha=0.5, tau=0.7, n_negatives=5, seed=seed,
    )
    results["FairIR"] = (
        evaluate_full_ranking(fairir_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST),
        evaluate_fairness(fairir_model, data["train"], data["test"], is_senior, data["n_users"], data["n_items"], k_list=K_LIST),
    )

    print("\n=== [3/3] AIR-Fair (instance-adaptive gate, group_col=age_group_code) ===")
    airfair_model, _, gate_info = train_airfair(
        data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
        group_col="age_group_code", n_factors=64, n_epochs=15,
        delta_base=0.1, alpha_base=0.5, tau=0.7, n_negatives=5, seed=seed,
        invert_gate=invert_gate,
    )
    print(f"\n[gate diagnostic] senior (50+) mean alpha_u: "
          f"{gate_info['alpha_u'][is_senior == 1].mean():.4f}")
    print(f"[gate diagnostic] non-senior mean alpha_u: "
          f"{gate_info['alpha_u'][is_senior == 0].mean():.4f}")
    results["AIR-Fair"] = (
        evaluate_full_ranking(airfair_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=K_LIST),
        evaluate_fairness(airfair_model, data["train"], data["test"], is_senior, data["n_users"], data["n_items"], k_list=K_LIST),
    )

    print("\n" + "=" * 65)
    print("=== FINAL COMPARISON (all K, fairness = 50+ vs rest) ===")
    for k in K_LIST:
        print(f"\n--- K={k} ---")
        print(f"{'Model':<10} {'HR':>10} {'NDCG':>10} {'DP':>10} {'EO':>10}")
        for model_name, (acc, fair) in results.items():
            print(f"{model_name:<10} {acc[f'HR@{k}']:>10.4f} {acc[f'NDCG@{k}']:>10.4f} "
                  f"{fair[f'DP@{k}']:>10.4f} {fair[f'EO@{k}']:>10.4f}")

    rows = []
    gate_mode_tag = "inverted" if invert_gate else "standard"
    for model_name, (acc, fair) in results.items():
        for k in K_LIST:
            rows.append({
                "dataset": dataset_name, "target_attribute": "age_50plus_vs_rest",
                "gate_mode": gate_mode_tag,
                "model": model_name, "K": k,
                "HR": acc[f"HR@{k}"], "NDCG": acc[f"NDCG@{k}"],
                "DP": fair[f"DP@{k}"], "EO": fair[f"EO@{k}"],
            })
    results_df = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, f"airfair_comparison_results_age_{gate_mode_tag}.csv")
    out_path = get_versioned_path(out_path)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--positive-threshold", type=float, default=3)
    parser.add_argument("--invert-gate", action="store_true",
                         help="Give disadvantaged users LESS cross-group noise (protect their "
                              "real signal) instead of more -- tests the opposite polarity.")
    args = parser.parse_args()
    pos_thresh = None if args.positive_threshold == 0 else args.positive_threshold
    run(dataset_name=args.dataset, positive_threshold=pos_thresh, invert_gate=args.invert_gate)