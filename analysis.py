"""Greedy-seeded simulated annealing vs random-restart hill climbing for test assembly
on 40 simulated item banks, referenced to an exact MILP optimum."""
import json, os, sys, math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- fixed classified criteria (identical for all banks) ----
N, NB, NI = 30, 40, 300                  # test length, banks, items per bank
TT = np.full(5, 6.0)                     # items per topic
TY = np.array([12.0, 10.0, 8.0])         # items per type
T, D = 3800.0, 0.55                      # total time target, mean difficulty target
W1, W2, WT, WD, WA = 1.0, 1.0, 20.0, 40.0, 10.0   # penalty weights
BUDGET = 200_000                         # penalty evaluations per run (both methods)
SEEDS = (42, 43, 44)


def make_bank(b):
    """Simulate one bank with topic, type, difficulty, discrimination, time."""
    r = np.random.default_rng(42 + b)
    topic, typ = r.integers(0, 5, NI), r.choice(3, NI, p=[.5, .3, .2])
    diff, a = r.beta(2, 2, NI), r.lognormal(0, .35, NI)
    tm = 70 + 80 * diff + r.uniform(0, 40, NI)
    return dict(topic=topic, typ=typ, diff=diff, a=a, tm=tm, oht=np.eye(5)[topic],
                ohy=np.eye(3)[typ], A=0.95 * np.sort(a)[-N:].mean())  # soft discrimination goal


def pen_vec(bk, ct, cy, ts, ds, as_, n):
    """Weighted-deviation penalty; targets are prorated by n/N for partial tests (greedy)."""
    f = n / N
    return (W1 * np.abs(ct - TT * f).sum(-1) + W2 * np.abs(cy - TY * f).sum(-1)
            + WT * np.abs(ts - T * f) / T + WD * np.abs(ds / n - D)
            + WA * np.maximum(0, bk["A"] - as_ / n))


def stats(bk, sel):
    return (bk["oht"][sel].sum(0), bk["ohy"][sel].sum(0), bk["tm"][sel].sum(),
            bk["diff"][sel].sum(), bk["a"][sel].sum())


def penalty(bk, sel):
    return float(pen_vec(bk, *stats(bk, sel), N))


def unmet(bk, sel):
    """Number of unmet criteria out of 11."""
    ct, cy, ts, ds, as_ = stats(bk, sel)
    return int((ct != TT).sum() + (cy != TY).sum() + (abs(ts - T) / T > .01)
               + (abs(ds / N - D) > .02) + (max(0, bk["A"] - as_ / N) > .02 * bk["A"]))


def hill_climb(bk, rng):
    """Random start, best single swap until local optimum, random restart until budget is spent."""
    oht, ohy, tm, df, a = bk["oht"], bk["ohy"], bk["tm"], bk["diff"], bk["a"]
    M = N * (NI - N)
    evals, best, best_sel = 0, math.inf, None
    while evals + 1 + M <= BUDGET:
        perm = rng.permutation(NI)
        sel, out = perm[:N].copy(), perm[N:].copy()
        cur = penalty(bk, sel); evals += 1
        if cur < best: best, best_sel = cur, sel.copy()
        while evals + M <= BUDGET:       # full neighbourhood = 30*270 evaluations
            ct, cy, ts, ds, as_ = stats(bk, sel)
            p = pen_vec(bk,
                        ct - oht[sel][:, None, :] + oht[out][None, :, :],
                        cy - ohy[sel][:, None, :] + ohy[out][None, :, :],
                        ts - tm[sel][:, None] + tm[out][None, :],
                        ds - df[sel][:, None] + df[out][None, :],
                        as_ - a[sel][:, None] + a[out][None, :], N)
            evals += M
            i, j = np.unravel_index(np.argmin(p), p.shape)
            if p[i, j] >= cur - 1e-12: break   # local optimum
            sel[i], out[j] = out[j], sel[i]
            cur = float(p[i, j])
            if cur < best: best, best_sel = cur, sel.copy()
    return best_sel


def greedy(bk):
    """Add the item that minimises the (prorated) penalty; returns selection and evals used."""
    ct, cy, ts, ds, as_ = np.zeros(5), np.zeros(3), 0.0, 0.0, 0.0
    avail, sel, evals = np.ones(NI, bool), [], 0
    for k in range(1, N + 1):
        c = np.flatnonzero(avail)
        p = pen_vec(bk, ct + bk["oht"][c], cy + bk["ohy"][c], ts + bk["tm"][c],
                    ds + bk["diff"][c], as_ + bk["a"][c], k)
        evals += len(c)
        i = c[np.argmin(p)]
        avail[i] = False; sel.append(int(i))
        ct, cy = ct + bk["oht"][i], cy + bk["ohy"][i]
        ts, ds, as_ = ts + bk["tm"][i], ds + bk["diff"][i], as_ + bk["a"][i]
    return sel, evals


def sim_anneal(bk, rng):
    """Greedy seed, then simulated annealing over single-item swaps (rest of the budget)."""
    sel, evals = greedy(bk)
    out = [i for i in range(NI) if i not in set(sel)]
    tp, ty, tm, df, a = (bk[k].tolist() for k in ("topic", "typ", "tm", "diff", "a"))
    A = bk["A"]
    ct, cy, ts, ds, as_ = stats(bk, np.array(sel))
    ct, cy = [int(x) for x in ct], [int(x) for x in cy]
    TTl, TYl = TT.tolist(), TY.tolist()

    def pen_s(ct, cy, ts, ds, as_):      # scalar twin of pen_vec for n = N
        return (W1 * sum(abs(c - t) for c, t in zip(ct, TTl)) + W2 * sum(abs(c - t) for c, t in zip(cy, TYl))
                + WT * abs(ts - T) / T + WD * abs(ds / N - D) + WA * max(0.0, A - as_ / N))

    cur = pen_s(ct, cy, ts, ds, as_)
    best, best_sel = cur, sel[:]
    iters = BUDGET - evals
    t0, t1 = 0.5, 0.01                   # fixed geometric cooling schedule
    alpha, temp = (t1 / t0) ** (1 / iters), t0
    P, Q, U = (rng.integers(0, N, iters).tolist(), rng.integers(0, NI - N, iters).tolist(),
               rng.random(iters).tolist())
    for p, q, u in zip(P, Q, U):
        i, j = sel[p], out[q]
        ct[tp[i]] -= 1; ct[tp[j]] += 1; cy[ty[i]] -= 1; cy[ty[j]] += 1
        nts, nds, nas = ts - tm[i] + tm[j], ds - df[i] + df[j], as_ - a[i] + a[j]
        new = pen_s(ct, cy, nts, nds, nas)
        d = new - cur
        if d <= 0 or u < math.exp(-d / temp):
            sel[p], out[q], cur, ts, ds, as_ = j, i, new, nts, nds, nas
            if cur < best - 1e-12: best, best_sel = cur, sel[:]
        else:
            ct[tp[i]] += 1; ct[tp[j]] -= 1; cy[ty[i]] += 1; cy[ty[j]] -= 1
        temp *= alpha
    return np.array(best_sel)


def solve_milp(bk):
    """Exact optimum of the same penalty (linearised with deviation variables)."""
    nv = NI + 11                         # x, 5 topic dev, 3 type dev, time dev, diff dev, shortfall
    c = np.r_[np.zeros(NI), [W1] * 5, [W2] * 3, WT / T, WD, WA]
    rows, ub = [], []

    def add(coef, dev, const, signs=(1, -1)):
        for s in signs:                  # dev >= s * (coef.x - const)
            r = np.zeros(nv); r[:NI] = s * coef; r[NI + dev] = -1
            rows.append(r); ub.append(s * const)
    for k in range(5): add((bk["topic"] == k).astype(float), k, TT[k])
    for k in range(3): add((bk["typ"] == k).astype(float), 5 + k, TY[k])
    add(bk["tm"], 8, T)
    add(bk["diff"] / N, 9, D)
    add(bk["a"] / N, 10, bk["A"], signs=(-1,))   # shortfall only
    cons = [LinearConstraint(np.array(rows), -np.inf, np.array(ub)),
            LinearConstraint(np.r_[np.ones(NI), np.zeros(11)][None, :], N, N)]
    res = milp(c, constraints=cons, integrality=np.r_[np.ones(NI), np.zeros(11)],
               bounds=Bounds(np.zeros(nv), np.r_[np.ones(NI), [np.inf] * 11]),
               options={"time_limit": 10, "mip_rel_gap": 1e-6})
    if res.x is None: return None, False
    sel = np.flatnonzero(res.x[:NI] > .5)
    return (penalty(bk, sel), res.status == 0) if len(sel) == N else (None, False)


def run_bank(b):
    bk = make_bank(b)
    opt, proven = solve_milp(bk)
    if opt is None: return None
    out = dict(opt=opt, proven=proven)
    for name, fn in (("hc", hill_climb), ("sa", sim_anneal)):
        sels = [np.asarray(fn(bk, np.random.default_rng([s, b]))) for s in SEEDS]
        out[name] = [(penalty(bk, s), unmet(bk, s)) for s in sels]
    return out


if __name__ == "__main__":
    print("Solving MILP and running both methods on", NB, "banks ...", flush=True)
    res = [r for r in Parallel(n_jobs=-1)(delayed(run_bank)(b) for b in range(NB)) if r]
    print("valid banks:", len(res), flush=True)
    if len(res) < 30: sys.exit("too few banks with an MILP reference")
    opt = np.array([r["opt"] for r in res])
    m = {}
    for name, key in (("baseline", "hc"), ("improved", "sa")):
        pen = np.array([[x[0] for x in r[key]] for r in res])      # banks x seeds
        unm = np.array([[x[1] for x in r[key]] for r in res], float)
        gap = (pen - opt[:, None]) / np.maximum(opt[:, None], 1e-3)
        for tag, v in (("score", gap.mean(0)), ("within1pct", (gap <= .01).mean(0)), ("unmet", unm.mean(0))):
            m[f"{name}_{tag}"], m[f"{name}_{tag}_std"] = float(v.mean()), float(v.std())
        print(name, "mean rel. gap per seed:", np.round(gap.mean(0), 4), flush=True)
    m["improvement"] = m["baseline_score"] - m["improved_score"]
    m.update(n_valid_banks=len(res), share_milp_proven_optimal=float(np.mean([r["proven"] for r in res])),
             mean_optimal_penalty=float(opt.mean()), eval_budget_per_run=BUDGET)
    if not all(np.isfinite(v) for v in m.values()): sys.exit("non-finite metrics")
    os.makedirs("figures", exist_ok=True)
    plt.bar(["hill climbing", "greedy+SA"], [m["baseline_score"], m["improved_score"]],
            yerr=[m["baseline_score_std"], m["improved_score_std"]], color=["gray", "tab:blue"])
    plt.ylabel("mean relative gap to optimum (lower is better)")
    plt.tight_layout(); plt.savefig("figures/gap.png", dpi=120)
    out = {
        "question": "Can a greedy-seeded simulated-annealing swap search come closer to the proven-optimal test than a plain hill-climbing assembler under the same evaluation budget on simulated item banks with classified criteria?",
        "method": "40 seeded simulated item banks (300 items); weighted-deviation penalty over topic/type counts, total time, mean difficulty and a soft discrimination goal; both heuristics get the same penalty-evaluation budget and run with seeds 42-44; reference optimum from an exact MILP (HiGHS).",
        "dataset": {"name": "Simulated item banks with classified criteria", "source_url": "generated in analysis.py (numpy)",
                    "n_rows": NB * NI, "n_train": 0, "n_test": NB, "same_as_paper": False,
                    "baseline_is_papers_method": False},
        "metric_name": "mean relative gap to optimal penalty", "higher_is_better": False,
        "baseline": "Standard hill-climbing assembler: random start, best single-item swap, stop at local optimum, random restarts until the evaluation budget is spent (stand-in; the paper's modifications are unknown).",
        "improved": "Greedy construction as seed followed by simulated annealing over single-item swaps with a fixed geometric cooling schedule, using the same evaluation budget (greedy evaluations are charged to it).",
        "metrics": m,
        "notes": "No training/tuning: the 40 banks are the test set and SA settings (T 0.5->0.01) were fixed a priori. Banks, criteria and weights are synthetic, so results may not transfer to real banks. Relative gap = (penalty - MILP penalty)/max(MILP penalty, 1e-3); MILP has a 10 s limit per bank, so the reference is proven optimal only for the share given in share_milp_proven_optimal (gaps can be negative otherwise). 'Unmet' counts 11 criteria with tolerances (time 1%, difficulty 0.02, discrimination 2% of goal).",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(m, indent=2))
