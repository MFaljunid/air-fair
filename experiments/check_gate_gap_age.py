"""
check_gate_gap_age.py — same diagnostic as check_gate_gap.py, but using
AGE_GROUP instead of gender. Motivated by the earlier CD-AFMS finding that
the "50+" age group showed a stronger statistical signal than gender on
ML-1M. Since all our AIR-Fair code already accepts group_col as a parameter
(no hardcoded binary assumption), this required zero code changes -- just a
different group_col.

Results are saved to results/<dataset>/diagnostics/quality_gap_by_age.csv
so every diagnostic check we run stays documented, not just the "real"
experiment scripts.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from versioning import get_versioned_path

import numpy as np
import pandas as pd

from preprocessing.load_data import load_and_prepare
from preprocessing.attributes import prepare_all_attributes
from models.mf import train_bpr
from fairness.instance_gate import compute_user_quality_proxy, compute_gate_values

DATASET_NAME = "ml-1m"
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_DIR = os.path.join(project_root, "data", "raw", DATASET_NAME)
DIAGNOSTICS_DIR = os.path.join(project_root, "results", DATASET_NAME, "diagnostics")
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

data = load_and_prepare(raw_dir=RAW_DIR, positive_threshold=3, seed=0)
attr_result = prepare_all_attributes(data["user_attrs"])
user_attrs_sorted = attr_result["user_attrs"].sort_values("user_id").reset_index(drop=True)
age_group = user_attrs_sorted["age_group_code"].values

model = train_bpr(
    data["train"], data["n_users"], data["n_items"],
    n_factors=64, n_epochs=15, seed=0,
)

quality = compute_user_quality_proxy(model, data["train"], data["n_users"], data["n_items"], n_samples_per_user=50, seed=0)
g_values, dominant_group, dominant_quality = compute_gate_values(quality, age_group)

age_labels = {0: "<25", 1: "25-34", 2: "35-49", 3: "50+"}
overall_std = np.nanstd(quality)

print(f"\nDominant age group: {age_labels.get(dominant_group, dominant_group)}, quality={dominant_quality:.4f}")

rows = []
for g in sorted(set(age_group)):
    mask = age_group == g
    mean_q = np.nanmean(quality[mask])
    std_q = np.nanstd(quality[mask])
    mean_g = np.nanmean(g_values[mask])
    relative_gap = abs(mean_q - dominant_quality) / overall_std
    is_dominant = (g == dominant_group)

    print(f"Age {age_labels.get(g, g):<6}: n={mask.sum():>5}, mean quality={mean_q:.4f}, "
          f"std={std_q:.4f}, mean g(u)={mean_g:.4f}, relative_gap={relative_gap:.4f}"
          f"{' (dominant)' if is_dominant else ''}")

    rows.append({
        "dataset": DATASET_NAME, "attribute": "age_group", "group": age_labels.get(g, g),
        "n_users": int(mask.sum()), "mean_quality": mean_q, "std_quality": std_q,
        "mean_gate_g": mean_g, "relative_gap_vs_dominant": relative_gap, "is_dominant": is_dominant,
    })

results_df = pd.DataFrame(rows)
out_path = os.path.join(DIAGNOSTICS_DIR, "quality_gap_by_age.csv")
out_path = get_versioned_path(out_path)
results_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")