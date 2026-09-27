"""
tune_gccf_lr.py — throwaway: sweep a few (lr, accumulation_steps) combos
DIRECTLY on real ml-1m data (only 5 epochs each, fast) to find a setting
that's actually stable on this dataset's real (skewed) degree distribution,
rather than trusting values tuned on synthetic uniform-random test data.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from preprocessing.load_data import load_and_prepare
from models.gccf import train_gccf_bpr

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "ml-1m")
data = load_and_prepare(raw_dir=RAW_DIR, positive_threshold=3, seed=0)
print(f"n_users={data['n_users']}, n_items={data['n_items']}, train={len(data['train'])}\n")

combos = [
    {"lr": 0.05, "accumulation_steps": 50},
    {"lr": 0.1,  "accumulation_steps": 50},
    {"lr": 0.5,  "accumulation_steps": 50},
    {"lr": 0.1,  "accumulation_steps": 200},
    {"lr": 0.05, "accumulation_steps": 10},
]

for combo in combos:
    print(f"=== lr={combo['lr']}, accumulation_steps={combo['accumulation_steps']} ===")
    model = train_gccf_bpr(
        data["train"], data["n_users"], data["n_items"],
        n_factors=64, n_layers=2, n_epochs=5, seed=0, verbose=True, **combo,
    )
    norm_max = np.linalg.norm(model.user_emb, axis=1).max()
    print(f"  -> embedding norm max: {norm_max:.4f}  {'DIVERGED' if norm_max > 10 else 'stable'}\n")