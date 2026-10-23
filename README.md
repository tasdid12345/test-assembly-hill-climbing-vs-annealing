# Greedy-seeded simulated annealing vs hill climbing for test assembly

## Motivation
The paper "A modified hill-climbing algorithm for knowledge test assembly based on classified criteria" (2016) proposes selecting test items from an item bank according to formal, classified criteria. Only the abstract was available, so its exact modifications are unknown. This experiment checks whether a different local-search design gets closer to the exact optimum than a standard hill-climbing assembler.

## Question
Can a greedy-seeded simulated-annealing swap search come closer to the proven-optimal test than a plain hill-climbing assembler under the same evaluation budget on simulated item banks with classified criteria?

## Dataset
40 seeded simulated item banks of 300 items each (topic, item type, difficulty, discrimination, time), generated in `analysis.py`. They are not the paper's data. Criteria: items per topic, items per type, total time, mean-difficulty target and a soft discrimination goal, combined into one weighted-deviation penalty. The 40 banks form the test set; nothing is trained or tuned.

## Baseline vs improved
- **Baseline:** standard hill climbing (random start, best single-item swap, stop at a local optimum, random restarts until the budget is spent). It is a stand-in, not the paper's method.
- **Improved:** greedy construction as a seed, then simulated annealing over single-item swaps with the same evaluation budget.
- **Reference:** exact MILP (`scipy.optimize.milp`, HiGHS, time-limited).
- **Metrics:** mean relative gap to the optimal penalty (main), share of banks within 1% of optimal, and mean number of unmet criteria. Heuristics run with seeds 42, 43 and 44.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On 40 simulated item banks (300 items each), a standard random-restart hill-climbing assembler had a mean relative gap to optimal penalty of 96.1% (SD 6.1, lower is better), while greedy-seeded simulated annealing reached 48.3% (SD 5.0) under the same evaluation budget.

| Metric | Value |
|---|---|
| baseline_score | 96.11 |
| baseline_score_std | 6.054 |
| baseline_within1pct | 0 |
| baseline_within1pct_std | 0 |
| baseline_unmet | 1.583 |
| baseline_unmet_std | 0.03118 |
| improved_score | 48.29 |
| improved_score_std | 5.044 |
| improved_within1pct | 0 |
| improved_within1pct_std | 0 |
| improved_unmet | 1.475 |
| improved_unmet_std | 0.03536 |
| improvement | 47.81 |
| n_valid_banks | 40 |
| share_milp_proven_optimal | 0.85 |
| mean_optimal_penalty | 0.228 |
| eval_budget_per_run | 200000 |

Run on 2026-10-23. Full output: [results.json](results.json) · [run.log](run.log)

Notes: No training/tuning: the 40 banks are the test set and SA settings (T 0.5->0.01) were fixed a priori. Banks, criteria and weights are synthetic, so results may not transfer to real banks. Relative gap = (penalty - MILP penalty)/max(MILP penalty, 1e-3); MILP has a 10 s limit per bank, so the reference is proven optimal only for the share given in share_milp_proven_optimal (gaps can be negative otherwise). 'Unmet' counts 11 criteria with tolerances (time 1%, difficulty 0.02, discrimination 2% of goal).

## Reference

This project is motivated by the following paper:

> DM Bojić, AM Bošnjaković, JŽ Protić (2016). *A modified hill-climbing algorithm for knowledge test assembly based on classified criteria*. International Journal of. https://www.worldscientific.com/doi/abs/10.1142/S0218194016500327

Paper author: Prof. Jelica Protić, University of Belgrade (jeca@etf.bg.ac.rs)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
