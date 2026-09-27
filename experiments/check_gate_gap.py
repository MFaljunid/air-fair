"""
check_gate_gap.py — diagnostic: is there a real quality gap between gender
groups on ML-1M, detectable by our instance-gate proxy? Trains plain BPR
(same as always) then inspects the gate values it would produce, WITHOUT
running full AIR-Fair training -- cheap, fast, answers the key question.

Results are saved to results/<dataset>/diagnostics/quality_gap_by_gender.csv
so every diagnostic check we run stays documented, not just the "real"
experiment scripts.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _versioning import get_versioned_path

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
user_group = user_attrs_sorted["gender_code"].values

model = train_bpr(
    data["train"], data["n_users"], data["n_items"],
    n_factors=64, n_epochs=15, seed=0,
)

quality = compute_user_quality_proxy(model, data["train"], data["n_users"], data["n_items"], n_samples_per_user=50, seed=0)
g_values, dominant_group, dominant_quality = compute_gate_values(quality, user_group)

overall_std = np.nanstd(quality)
gender_labels = {0: "M", 1: "F"}

print(f"\nDominant group: {gender_labels.get(dominant_group, dominant_group)}, quality={dominant_quality:.4f}")

rows = []
for g in [0, 1]:
    mask = user_group == g
    mean_q = np.nanmean(quality[mask])
    std_q = np.nanstd(quality[mask])
    mean_g = np.nanmean(g_values[mask])
    relative_gap = abs(mean_q - dominant_quality) / overall_std
    is_dominant = (g == dominant_group)

    print(f"Group {gender_labels[g]}: n={mask.sum()}, mean quality={mean_q:.4f}, "
          f"std={std_q:.4f}, mean g(u)={mean_g:.4f}, relative_gap={relative_gap:.4f}"
          f"{' (dominant)' if is_dominant else ''}")

    rows.append({
        "dataset": DATASET_NAME, "attribute": "gender", "group": gender_labels[g],
        "n_users": int(mask.sum()), "mean_quality": mean_q, "std_quality": std_q,
        "mean_gate_g": mean_g, "relative_gap_vs_dominant": relative_gap, "is_dominant": is_dominant,
    })

results_df = pd.DataFrame(rows)
out_path = os.path.join(DIAGNOSTICS_DIR, "quality_gap_by_gender.csv")
out_path = get_versioned_path(out_path)
results_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print(f"\nOverall quality distribution: min={np.nanmin(quality):.4f}, max={np.nanmax(quality):.4f}, std={overall_std:.4f}")