"""
check_fairir_threshold.py — one-off diagnostic script (throwaway) to verify
positive_threshold=3 brings our DP/EO numbers closer to FairIR's reported
BPR baseline on MovieLens-1M (DP@10=0.6541, EO@10=0.7158, 513112 positive
interactions). Run from the project root: py experiments/check_fairir_threshold.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from preprocessing.load_data import load_and_prepare
from preprocessing.attributes import prepare_all_attributes
from models.mf import train_bpr
from eval.metrics import evaluate_full_ranking, evaluate_fairness

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "ml-1m")

data = load_and_prepare(raw_dir=RAW_DIR, positive_threshold=3, seed=0)
print(f"n_users={data['n_users']}, n_items={data['n_items']}, train={len(data['train'])}")

model = train_bpr(
    data["train"], data["n_users"], data["n_items"],
    n_factors=64, n_epochs=15, seed=0,
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

print()
print("=== Results ===")
print(acc)
print(fair)
print()
print("FairIR reported (Table 3, BPR baseline, K=10): DP=0.6541, EO=0.7158")
print("FairIR reported total positive interactions: 513112")