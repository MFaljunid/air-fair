"""
compare_splits.py — quick diagnostic to isolate whether the DP/EO gap vs
FairIR's reported numbers is caused by the LOO vs random-80/20 protocol
difference, or something else. Run once, throwaway script (not part of the
final pipeline).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from preprocessing.load_data import load_and_prepare
from preprocessing.attributes import prepare_all_attributes
from models.mf import train_bpr
from eval.metrics import evaluate_full_ranking, evaluate_fairness

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "ml-1m")

for split in ["leave_one_out", "random"]:
    print(f"=== split_method={split} ===")
    data = load_and_prepare(raw_dir=RAW_DIR, split_method=split, seed=0)
    model = train_bpr(
        data["train"], data["n_users"], data["n_items"],
        n_factors=64, n_epochs=15, seed=0, verbose=False,
    )
    acc = evaluate_full_ranking(
        model, data["train"], data["test"], data["n_users"], data["n_items"], k_list=[10]
    )
    attr_result = prepare_all_attributes(data["user_attrs"])
    user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)
    user_group = user_attrs_sorted["gender_code"].values

    fair = evaluate_fairness(
        model, data["train"], data["test"], user_group,
        data["n_users"], data["n_items"], k_list=[10],
    )
    print(f"HR@10={acc['HR@10']:.4f}  NDCG@10={acc['NDCG@10']:.4f}  "
          f"DP@10={fair['DP@10']:.4f}  EO@10={fair['EO@10']:.4f}")
    print()

print("FairIR's own reported BPR baseline (Table 3, K=10): DP=0.6541, EO=0.7158")
