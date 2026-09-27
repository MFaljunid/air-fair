"""
quick_gccf_sanity.py — throwaway, FAST (~30s) sanity check: trains GCCF for
only 5 epochs directly on the REAL ml-1m data, to confirm (or rule out)
whether the fix is actually being exercised, without waiting through the
full 10-minute 4-model comparison.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from preprocessing.load_data import load_and_prepare
from models.gccf import train_gccf_bpr

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "ml-1m")
data = load_and_prepare(raw_dir=RAW_DIR, positive_threshold=3, seed=0)
print(f"n_users={data['n_users']}, n_items={data['n_items']}, train={len(data['train'])}")

model = train_gccf_bpr(
    data["train"], data["n_users"], data["n_items"],
    n_factors=64, n_layers=2, n_epochs=5, seed=0,
)
print("\nany NaN:", np.isnan(model.user_emb).any())
print("embedding norm max:", np.linalg.norm(model.user_emb, axis=1).max())