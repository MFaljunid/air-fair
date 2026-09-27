"""
diagnostic_ml1m_convergence.py
----------------------------------
Captures the epoch-by-epoch training curves on ML-1M at the STRONG
setting (alpha=0.8, delta=0.5, 30 epochs) -- the same setting used
everywhere else in the paper -- so Figure 5 (convergence) can show a real
ML-1M panel alongside the existing LastFM one, instead of ML-1M's old
weak-setting (alpha=0.5, delta=0.1, 15 epoch) curve from earlier today.

Just re-runs plain BPR and FairIR with verbose=True; copy the printed
per-epoch lines into make_figures2.py's ml1m_bpr_loss / ml1m_fairir_bpr_loss
/ ml1m_fairir_contrastive_loss arrays afterward.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from preprocessing.load_data import load_and_prepare
from preprocessing.attributes import prepare_all_attributes
from models.mf import train_bpr
from training.train_fairir import train_fairir

SEED = 0
N_EPOCHS = 30
STRONG_ALPHA = 0.8
STRONG_DELTA = 0.5

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
raw_dir = os.path.join(project_root, "data", "raw", "ml-1m")

print("=== Loading ML-1M ===")
data = load_and_prepare(raw_dir=raw_dir, positive_threshold=3, seed=SEED)
attr_result = prepare_all_attributes(data["user_attrs"])

print("\n=== [1/2] Plain BPR (verbose) ===")
train_bpr(data["train"], data["n_users"], data["n_items"],
          n_factors=64, n_epochs=N_EPOCHS, seed=SEED, verbose=True)

print("\n=== [2/2] FairIR (strong, verbose) ===")
train_fairir(data["train"], attr_result["user_attrs"], data["n_users"], data["n_items"],
             group_col="gender_code", n_factors=64, n_epochs=N_EPOCHS,
             delta=STRONG_DELTA, alpha=STRONG_ALPHA, tau=0.7, n_negatives=5, seed=SEED, verbose=True)

print("\nDone -- copy the per-epoch lines above into the figure script.")