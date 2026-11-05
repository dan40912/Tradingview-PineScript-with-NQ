"""B3: single-feature rank-IC tables per period, with a permutation null for "how many would
pass by luck".  Uses tasks.pkl written by study.py.  Periods: 2024, 2025H1 (discovery),
2025H2 (validation, used only to check features that passed on discovery).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from study import meta_signed

OUT = os.path.join(HERE, "results")
tasks, cols = pd.read_pickle(os.path.join(OUT, "tasks.pkl"))


def period(date):
    return np.where(date < "2025-01-01", "2024", np.where(date <= "2025-06-30", "2025H1", "2025H2"))


def rank_ic(F, y):
    """F: (n, k) feature matrix with NaN; y: (n,). Spearman per column with pairwise-complete rows."""
    out = np.full(F.shape[1], np.nan); nn = np.zeros(F.shape[1], int)
    yr = pd.Series(y)
    for j in range(F.shape[1]):
        m = np.isfinite(F[:, j]) & np.isfinite(y)
        if m.sum() < 20:
            continue
        a = pd.Series(F[m, j]).rank().values; b = yr[m].rank().values
        out[j] = np.corrcoef(a, b)[0, 1]; nn[j] = m.sum()
    return out, nn


def tstat(ic, n):
    return ic * np.sqrt(np.maximum(n - 2, 1)) / np.sqrt(np.maximum(1 - ic ** 2, 1e-9))


# targets: each fixed-time task (fwd return for dir tasks, binary for hit tasks)
targets = {}
for k, d in tasks.items():
    if d.kind.iloc[0] == "dir":
        targets[k] = (d, d.fwd.values.astype(float), d[cols].values.astype(float))
    elif d.kind.iloc[0] == "hit":
        targets[k] = (d, d.y.values.astype(float), d[cols].values.astype(float))
dm = tasks["meta_i2"]
targets["meta_i2"] = (dm, dm.pnl.values.astype(float), meta_signed(dm, cols).values.astype(float))


def ic_table(perm_rng=None):
    rows = []
    for k, (d, y, F) in targets.items():
        per = period(d.date.values)
        for p in ["2024", "2025H1", "2025H2"]:
            m = per == p
            yy = y[m].copy()
            if perm_rng is not None:
                yy = perm_rng.permutation(yy)
            ic, n = rank_ic(F[m], yy)
            rows.append(pd.DataFrame({"target": k, "period": p, "feature": cols, "ic": ic, "n": n}))
    T = pd.concat(rows)
    W = T.pivot_table(index=["target", "feature"], columns="period", values="ic")
    N = T.pivot_table(index=["target", "feature"], columns="period", values="n")
    return W, N


def passes(W, N):
    t24, t25 = tstat(W["2024"], N["2024"]), tstat(W["2025H1"], N["2025H1"])
    same = np.sign(W["2024"]) == np.sign(W["2025H1"])
    A = same & (t24.abs() >= 2) & (t25.abs() >= 2)
    B = same & (W["2024"].abs() >= 0.05) & (W["2025H1"].abs() >= 0.05)
    return same, A, B


W, N = ic_table()
same, A, B = passes(W, N)
W["t2024"] = tstat(W["2024"], N["2024"]).round(2); W["t2025H1"] = tstat(W["2025H1"], N["2025H1"]).round(2)
W["t2025H2"] = tstat(W["2025H2"], N["2025H2"]).round(2)
W["same_sign_disc"] = same; W["passA"] = A; W["passB"] = B
W["val_same_sign"] = np.sign(W["2025H2"]) == np.sign(W["2024"])
W.round(4).to_csv(os.path.join(OUT, "ic_table.csv"))

ntest = int(np.isfinite(W["2024"]).sum())
obs = {"n_tests": ntest, "same_sign": int(same.sum()), "passA": int(A.sum()), "passB": int(B.sum()),
       "passA_val_same": int((A & W.val_same_sign).sum()), "passB_val_same": int((B & W.val_same_sign).sum())}
# permutation null (same day-permutation applied across all features of a target/period)
rng = np.random.default_rng(1)
null = []
for r in range(200):
    Wp, Np = ic_table(rng)
    s, a, b = passes(Wp, Np)
    null.append((int(s.sum()), int(a.sum()), int(b.sum())))
null = np.array(null)
summary = pd.DataFrame({
    "observed": [obs["same_sign"], obs["passA"], obs["passB"]],
    "null_mean": null.mean(0).round(1), "null_p95": np.quantile(null, 0.95, axis=0),
    "p_value": [(null[:, k] >= v).mean() for k, v in enumerate([obs["same_sign"], obs["passA"], obs["passB"]])]},
    index=["same sign 2024 & 2025H1", "A: same sign & |t|>=2 both", "B: same sign & |IC|>=0.05 both"])
summary.to_csv(os.path.join(OUT, "ic_summary.csv"))
print(obs); print(summary)
# per-target breakdown
bt = pd.DataFrame({"tests": W.groupby(level=0).size(), "passA": W.passA.groupby(level=0).sum(),
                   "passB": W.passB.groupby(level=0).sum(),
                   "passB_and_val_same": (W.passB & W.val_same_sign).groupby(level=0).sum()})
bt.to_csv(os.path.join(OUT, "ic_by_target.csv")); print(bt)
print(W[W.passA].sort_values("t2024").to_string())
