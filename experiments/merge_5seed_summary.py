"""
merge_5seed_summary.py
-------------------------
Combines the original alpha_min=0.05 run (seeds 0,1, saved in
airfair_wide_range.csv) with the extra seeds 2,3,4 run
(airfair_5seeds_extra_2_3_4.csv) into a single 5-seed cross-seed summary --
the final, decisive check on whether AIR-Fair's win-win pattern holds up
with proper statistical power, or was a 2-seed coincidence.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ml-1m")

ORIGINAL_FILE = "airfair_wide_range.csv"          # seeds 0,1 (alpha_min=0.05)
EXTRA_FILE = "airfair_5seeds_extra_2_3_4.csv"      # seeds 2,3,4 (alpha_min=0.05)

K_LIST = [10, 20, 30, 40]


def main():
    original_path = os.path.join(RESULTS_DIR, ORIGINAL_FILE)
    extra_path = os.path.join(RESULTS_DIR, EXTRA_FILE)

    if not os.path.exists(original_path):
        print(f"ERROR: {original_path} not found. Adjust ORIGINAL_FILE at the top of this script "
              f"to match your actual seeds-0,1 alpha_min=0.05 results file name.")
        return
    if not os.path.exists(extra_path):
        print(f"ERROR: {extra_path} not found. Run check_airfair_5seeds_extra.py first.")
        return

    df_original = pd.read_csv(original_path)
    df_extra = pd.read_csv(extra_path)
    combined = pd.concat([df_original, df_extra], ignore_index=True)

    print(f"Combined {combined['seed'].nunique()} seeds: {sorted(combined['seed'].unique())}")
    print("=" * 70)

    for k in K_LIST:
        sub = combined[combined["K"] == k]
        fairir_hr = sub["FairIR_HR_change_pct"].values
        airfair_hr = sub["AIRFair_HR_change_pct"].values
        fairir_dp = sub["FairIR_DP_reduction_pct"].values
        airfair_dp = sub["AIRFair_DP_reduction_pct"].values

        hr_recovered_per_seed = airfair_hr - fairir_hr
        dp_diff_per_seed = airfair_dp - fairir_dp

        print(f"\nK={k}  (n={len(sub)} seeds)")
        print(f"  FairIR:    mean HR change={fairir_hr.mean():+.2f}%  mean DP reduction={fairir_dp.mean():+.2f}%")
        print(f"  AIR-Fair:  mean HR change={airfair_hr.mean():+.2f}%  mean DP reduction={airfair_dp.mean():+.2f}%")
        print(f"  HR recovered per seed: {hr_recovered_per_seed.round(2).tolist()}  "
              f"mean={hr_recovered_per_seed.mean():+.2f}pp  std={hr_recovered_per_seed.std():.2f}pp")
        print(f"  DP gain diff per seed: {dp_diff_per_seed.round(2).tolist()}  "
              f"mean={dp_diff_per_seed.mean():+.2f}pp  std={dp_diff_per_seed.std():.2f}pp")

        hr_consistent = (hr_recovered_per_seed > 0).all() or (hr_recovered_per_seed < 0).all()
        dp_consistent = (dp_diff_per_seed > 0).all() or (dp_diff_per_seed < 0).all()
        win_win_mean = (hr_recovered_per_seed.mean() > 0) and (dp_diff_per_seed.mean() > 0)
        win_win_consistent = win_win_mean and hr_consistent and dp_consistent and (hr_recovered_per_seed > 0).all()

        verdict = (
            "*** ROBUST WIN-WIN (consistent across ALL 5 seeds) ***" if win_win_consistent
            else "win-win ON AVERAGE only, NOT consistent across all seeds" if win_win_mean
            else "no win-win"
        )
        print(f"  -> {verdict}")

    out_path = os.path.join(RESULTS_DIR, "final_5seed_combined.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nCombined data saved to {out_path}")


if __name__ == "__main__":
    main()
