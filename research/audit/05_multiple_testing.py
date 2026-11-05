"""Multiple-testing accounting: variant counts, Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014),
Bonferroni haircut on the selection period, and a Holm test on the validation half-year across ALL candidates that
were frozen before validation (incl. the ones that failed). Holdout never loaded.  Output: out_05_*.csv + stdout."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_common as A

g, lv = A.load("NQ")
C = A.all_candidates()
AG = os.path.join(A.RES, "agents")

# ---------------------------------------------------------------- 1) variant counts (as reported by each agent)
VAR = [("A1", 640, "variants_log 576 + explore 55 + 9"), ("A2", 190, "engine variants (+72 descriptive screens, 20 random controls)"),
       ("A3", 980, "discovery grid (+179 validation neighbours looked at, +26 stress)"), ("B1", 275, ""),
       ("B2", 558, ""), ("B3", 590, "trade rules (+1,494 single-feature IC tests, 120 ML models)"),
       ("C1", 3860, "incl. 3,420-cell robustness grid"), ("C2", 375, "210 engine + ~165 descriptive (+40 post-val looks)"),
       ("C3", 286, "")]
vc = pd.DataFrame(VAR, columns=["agent", "variants", "note"])
N_TOTAL = int(vc.variants.sum())
print(vc.to_string(index=False)); print("TOTAL reported variants (trade rules):", N_TOTAL,
      " (+ ~1,500 IC tests, + >1,000 rule combos of the pre-project history on 2024-2026)")

# ---------------------------------------------------------------- 2) empirical dispersion of discovery Sharpe across trials
logs = [("A1/variants_log.csv", "sharpe_ann"), ("A2/variant_log.csv", "d_sharpe_ann"), ("C2/variant_log.csv", "d_sharpe_ann"),
        ("C1/out_03_grid.csv", "D_sharpe"), ("C1/out_01_screen.csv", "D_sharpe"), ("A3/out_grid_pts.csv", "sharpe_ann"),
        ("A3/out_mgmt.csv", "sharpe_ann"), ("C3/out_01_es_grid_discovery.csv", "sharpe"),
        ("B3/results/simple_rules.csv", "disc_sharpe_ann")]
sh = []
for f, c in logs:
    try:
        x = pd.to_numeric(pd.read_csv(os.path.join(AG, f), on_bad_lines="skip")[c], errors="coerce").dropna()
        x = x[np.isfinite(x)]
        sh.append(x); print(f"  {f:38s} n={len(x):5d} mean={x.mean():5.2f} sd={x.std():4.2f} max={x.max():5.2f}")
    except Exception as ex:
        print("  skip", f, ex)
allsh = pd.concat(sh)
SD_ANN = float(allsh.std())
print(f"pooled discovery annualised Sharpe across {len(allsh)} logged trials: mean {allsh.mean():.2f}, sd {SD_ANN:.2f}")

EG = 0.5772156649


def sr0(V, N):
    """expected max of N iid Sharpe estimates with variance V (per-period units)"""
    if N <= 1:
        return 0.0
    return np.sqrt(V) * ((1 - EG) * st.norm.ppf(1 - 1 / N) + EG * st.norm.ppf(1 - 1 / (N * np.e)))


def dsr(x, N, V):
    x = np.asarray(x, float); T = len(x)
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = st.skew(x), st.kurtosis(x, fisher=False)
    z = (sr - sr0(V, N)) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return st.norm.cdf(z), sr


# ---------------------------------------------------------------- 3) DSR / Bonferroni on discovery (selection period) and 2024-25
rows = []
for name in C:
    tr = A.trades(name, C)
    for per, sp in (("disc", "discovery"), ("2024-25", None)):
        x = A.daily_vec(g, tr, sp).values
        T = len(x)
        rec = {"cand": name, "period": per, "T_days": T}
        psr0, sr = dsr(x, 1, 1 / T)
        rec["SR_ann"] = round(sr * np.sqrt(252), 2)
        rec["p_raw(SR>0)"] = round(1 - psr0, 4)
        for N in (10, 100, 1000, N_TOTAL):
            rec[f"DSR_null_N{N}"] = round(dsr(x, N, 1 / T)[0], 3)        # V = sampling variance under H0
        rec[f"DSR_emp_N1000"] = round(dsr(x, 1000, (SD_ANN / np.sqrt(252)) ** 2)[0], 3)
        rec[f"DSR_emp_N{N_TOTAL}"] = round(dsr(x, N_TOTAL, (SD_ANN / np.sqrt(252)) ** 2)[0], 3)
        rec["Bonf_p_agent"] = round(min(1, rec["p_raw(SR>0)"] * {a: v for a, v, _ in VAR}[name.split("/")[0]]), 3)
        rec["Bonf_p_all"] = round(min(1, rec["p_raw(SR>0)"] * N_TOTAL), 3)
        rows.append(rec)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(A.HERE, "out_05_dsr.csv"), index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(d.to_string(index=False))

# ---------------------------------------------------------------- 4) Holm on validation, all pre-validation-frozen candidates
mods = {a: A.load_agent(a) for a in A.AGENTS}
frozen = {}
for a, m in mods.items():
    for attr in ("CANDIDATES", "REJECTED", "FAILED_VALIDATION"):
        for k, fn in getattr(m, attr, {}).items():
            frozen[f"{a}/{k}"] = fn
POSTHOC = {"C2/C2_TE_OUTSIDE_NOI2", "B3/B3_MOM_OPEN_1000_X80"}   # chosen after seeing validation -> not a fair test
hrows = []
for name, fn in frozen.items():
    if name in POSTHOC:
        continue
    tr = A.trades(name, {**C, name: fn}, fn)
    x = A.daily_vec(g, tr, "validation").values
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    p = 1 - st.t.cdf(t, len(x) - 1)
    hrows.append({"cand": name, "val_days": len(x), "val_trades": int((tr.split == "validation").sum()),
                  "val_total": round(x.sum(), 0), "val_SR_ann": round(x.mean() / x.std(ddof=1) * np.sqrt(252), 2),
                  "t": round(t, 2), "p_one_sided": round(p, 4)})
h = pd.DataFrame(hrows).sort_values("p_one_sided").reset_index(drop=True)
m = len(h)
h["holm_threshold"] = [0.05 / (m - k) for k in range(m)]
rej, stop = [], False
for k, r in h.iterrows():
    ok = (not stop) and r.p_one_sided <= r.holm_threshold
    stop |= not ok
    rej.append(ok)
h["holm_reject_H0"] = rej
h["bonf_p"] = (h.p_one_sided * m).clip(upper=1).round(3)
h.to_csv(os.path.join(A.HERE, "out_05_holm_validation.csv"), index=False)
print(f"\nHolm (family = {m} candidates frozen before validation; H0: validation daily mean <= 0)")
print(h.to_string(index=False))
