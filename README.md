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

## Reference

This project is motivated by the following paper:

> DM Bojić, AM Bošnjaković, JŽ Protić (2016). *A modified hill-climbing algorithm for knowledge test assembly based on classified criteria*. International Journal of. https://www.worldscientific.com/doi/abs/10.1142/S0218194016500327

Paper author: Prof. Jelica Protić, University of Belgrade (jeca@etf.bg.ac.rs)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
