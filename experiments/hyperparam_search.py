"""
hyperparam_search.py
----------------------
Small, cheap hyperparameter search over FairIR's alpha (noise fusion ratio)
and tau (contrastive temperature), on real data. Each combination's result
is appended to results/<dataset>/hyperparam_search.csv IMMEDIATELY after it
finishes -- so if the run is interrupted partway, every combination tried so
far is already saved to disk, nothing is lost.

Combinations are hand-picked (not a full grid) to stay cheap: default is 4
combos, spanning FairIR's own reported good tau range (0.5-1.1, Section 5.1)
and a few alpha values around the paper's default region.
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

K_LIST = [10, 20, 30, 40]

DEFAULT_COMBOS = [
    {"alpha": 0.5, "tau": 0.7},   # our original baseline run, for reference
    {"alpha": 0.3, "tau": 0.7},
    {"alpha": 0.7, "tau": 0.7},
    {"alpha": 0.5, "tau": 1.0},
]


def append_result_row(out_path: str, row: dict):
    """Append one row to the CSV, creating it with a header if it doesn't exist yet."""
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(out_path)
    df_row.to_csv(out_path, mode="a", header=write_header, index=False)


def run_search(dataset_name: str = "ml-1m", positive_threshold: float = 3, combos: list = None, seed: int = 0):
    combos = combos or DEFAULT_COMBOS
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    raw_dir = os.path.join(project_root, "data", "raw", dataset_name)
    results_dir = os.path.join(project_root, "results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "hyperparam_search.csv")

    print(f"=== Loading {dataset_name} (positive_threshold={positive_threshold}) ===")
    data = load_and_prepare(raw_dir=raw_dir, positive_threshold=positive_threshold, seed=seed)
    print(f"n_users={data['n_users']}, n_items={data['n_items']}, train={len(data['train'])}")

    attr_result = prepare_all_attributes(data["user_attrs"])
    user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)
    user_group = user_attrs_sorted["gender_code"].values

    print("\n=== Reference: plain BPR (fixed, doesn't depend on alpha/tau) ===")
    bpr_model = train_bpr(
        data["train"], data["n_users"], data["n_items"],
        n_factors=64, n_epochs=15, seed=seed,
    )
    bpr_acc = evaluate_full_ranking(
        bpr_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=[10]
    )
    bpr_fair = evaluate_fairness(
        bpr_model, data["train"], data["test"], user_group,
        data["n_users"], data["n_items"], k_list=[10],
    )
    append_result_row(out_path, {
        "dataset": dataset_name, "model": "BPR", "alpha": None, "tau": None,
        "HR@10": bpr_acc["HR@10"], "NDCG@10": bpr_acc["NDCG@10"],
        "DP@10": bpr_fair["DP@10"], "EO@10": bpr_fair["EO@10"],
    })
    print(f"BPR reference -> HR@10={bpr_acc['HR@10']:.4f} DP@10={bpr_fair['DP@10']:.4f} "
          f"EO@10={bpr_fair['EO@10']:.4f}")

    for combo_idx, combo in enumerate(combos):
        alpha, tau = combo["alpha"], combo["tau"]
        print(f"\n=== Combo {combo_idx + 1}/{len(combos)}: alpha={alpha}, tau={tau} ===")

        fairir_model, noise_module = train_fairir(
            data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
            group_col="gender_code",
            n_factors=64, n_epochs=15,
            delta=0.1, alpha=alpha, tau=tau, n_negatives=5,
            seed=seed,
        )
        acc = evaluate_full_ranking(
            fairir_model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=[10]
        )
        fair = evaluate_fairness(
            fairir_model, data["train"], data["test"], user_group,
            data["n_users"], data["n_items"], k_list=[10],
        )

        row = {
            "dataset": dataset_name, "model": "FairIR", "alpha": alpha, "tau": tau,
            "HR@10": acc["HR@10"], "NDCG@10": acc["NDCG@10"],
            "DP@10": fair["DP@10"], "EO@10": fair["EO@10"],
        }
        append_result_row(out_path, row)
        print(f"  -> HR@10={acc['HR@10']:.4f} NDCG@10={acc['NDCG@10']:.4f} "
              f"DP@10={fair['DP@10']:.4f} EO@10={fair['EO@10']:.4f}")
        print(f"  (saved to {out_path})")

    print(f"\n=== Search complete. All results in {out_path} ===")
    print(pd.read_csv(out_path))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--positive-threshold", type=float, default=3)
    args = parser.parse_args()

    pos_thresh = None if args.positive_threshold == 0 else args.positive_threshold
    run_search(dataset_name=args.dataset, positive_threshold=pos_thresh)