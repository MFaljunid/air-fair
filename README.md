# AIR-Fair: Instance-Adaptive Fairness Gating for Accuracy-Fairness Trade-offs in Cross-Group Recommendation

AIR-Fair extends [FairIR](https://doi.org/10.1016/j.knosys.2026.115279) (Shi et al., KBS 2026) by replacing its fixed, group-level fairness-correction strength with a per-user, self-supervised, label-free gate. Instead of applying the same correction to every member of a sensitive-attribute group, AIR-Fair scales the correction according to how well each individual user is currently served relative to their dominant group.

## Key finding

AIR-Fair recovers accuracy lost under FairIR's strong fixed correction, by up to 3.9 percentage points in Hit Rate, across two datasets and three sensitive-attribute settings (binary gender, multi-valued age, and intersectional gender x age). A randomized-gate ablation confirms this recovery is attributable to the quality-aware gating mechanism rather than to user-specific variability alone.

The price of that recovery, however, is dataset-dependent: on MovieLens-1M it comes with little to no reduction in FairIR's fairness gain, while on LastFM-360K the same mechanism trades away a consistent part of the fairness gain for the additional accuracy. AIR-Fair does not eliminate the accuracy-fairness trade-off inherent to group-level correction; it relocates it to a more favorable, but not fixed, operating point.

## Repository structure

```
air-fair/
├── preprocessing/
│   ├── load_data.py        # Dataset loading, k-core filtering, LOO/random split
│   └── attributes.py        # Gender/age/intersectional group construction
├── models/
│   └── mf.py                 # BPR matrix factorization backbone
├── fairness/
│   ├── cross_group_noise.py     # FairIR's cross-group noise injection
│   ├── contrastive_loss.py      # FairIR's contrastive distillation term
│   └── instance_gate.py         # AIR-Fair's instance-adaptive gate
├── training/
│   ├── train_fairir.py       # FairIR training loop
│   └── train_airfair.py      # AIR-Fair training loop (with randomize_gate ablation option)
├── eval/
│   └── metrics.py            # HR@K, NDCG@K, DP@K, EO@K (full-ranking)
├── experiments/               # All experiment scripts (see below)
├── results/                   # CSV results, organized by dataset
└── data/raw/                  # Place downloaded datasets here (not tracked by git)
```

## Datasets

- **MovieLens-1M**: <https://grouplens.org/datasets/movielens/1m/>
- **LastFM-360K**: <http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html>

Download and extract into `data/raw/ml-1m/` and `data/raw/lastfm/` respectively. Datasets are not tracked in this repository; see `.gitignore`.

## Reproducing the main results

```bash
# ML-1M, gender, 5 seeds
python experiments/check_airfair_wide_range.py --seeds 0 1
python experiments/check_airfair_5seeds_extra.py --seeds 2 3 4

# ML-1M, age (3 seeds) / intersectional (3 seeds) / ablation (3 seeds)
python experiments/check_multigroup_contribution.py
python experiments/check_intersectional_contribution.py
python experiments/check_ablation_random_gate.py

# LastFM-360K (final recipe: min_user=40, min_item=10, subsample=15000)
python experiments/check_lastfm_airfair_gender.py
python experiments/check_lastfm_airfair_age.py
python experiments/check_lastfm_airfair_intersectional.py
python experiments/check_lastfm_ablation.py
```

Each script saves its results to `results/<dataset>/`, auto-versioning the output file if one already exists so nothing is overwritten.

## Key hyperparameters

| Component | Parameter | Value |
|---|---|---|
| Backbone (BPR) | latent factors / lr / batch size | 64 / 0.05 / 1024 |
| FairIR (strong) | alpha / delta / epochs | 0.8 / 0.5 / 30 |
| AIR-Fair gate | alpha_min / alpha_max | 0.05 / 1.0 |
| LastFM filtering | min_user / min_item / subsample | 40 / 10 / 15,000 |

Full details in the paper's Experimental Setup section.

## Status

This repository accompanies a manuscript in preparation. Code is provided for transparency and reproducibility of the reported results.