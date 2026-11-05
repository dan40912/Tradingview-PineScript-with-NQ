"""Phase E: one-time holdout evaluation, following research/HOLDOUT_PREREG.md.
Writes research/results_holdout/ (summary CSVs + trades)."""
import sys, os, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats as st
from lib import data, features, sim, metrics, strategies

OUT = os.path.join(HERE, "results_holdout"); os.makedirs(os.path.join(OUT, "trades"), exist_ok=True)
G = {s: data.load_grid(s, include_holdout=True) for s in ("NQ", "ES")}
LV = {s: features.daily_levels(G[s]) for s in G}
g, lv = G["NQ"], LV["NQ"]
ND = pd.Series(g.split[g.tradable]).value_counts().to_dict()
HOLD_DAYS = [d for d, t in zip(g.dates, g.tradable) if t and data.split_of(d) == "holdout"]


def load(agent):
    p = os.path.join(HERE, "agents", agent, "candidates.py")
    sys.path.insert(0, os.path.dirname(p))
    spec = importlib.util.spec_from_file_location(f"c_{agent}", p); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); sys.path.pop(0); return m


def worst_month(tr):
    if len(tr) == 0: return 0.0
    return float(tr.groupby(tr.date.str[:7]).pnl.sum().min())


def row(name, tr, kind):
    out = []
    for sp in ["discovery", "validation", "holdout"]:
        t = tr[tr.split == sp]
        s = metrics.summary(t, n_days=ND.get(sp)) if len(t) else {"n": 0}
        if len(t):
            s["worst_day"] = round(t.groupby("date").pnl.sum().min(), 0); s["worst_month"] = round(worst_month(t), 0)
        s.update(candidate=name, kind=kind, split=sp); out.append(s)
    return out


base = sim.run_signals(g, strategies.i2_v1_signals(g, lv)); base["symbol"] = "NQ"
rows = row("BASE/I2v1", base, "benchmark")
trades = {"BASE/I2v1": base}
PRIMARY = {"E1": ("A3", "A3_C3_POC_SL60_NP30"), "E2": ("C1", "C1_ORB15_TP110_TS"),
           "E3": ("C2", "C2_TE_OUTSIDE"), "E4": ("C3", "C3_ES_I2_40_15")}
mods = {a: load(a) for a in ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]}
prim_keys = {f"{a}/{n}" for a, n in PRIMARY.values()}
for a, m in mods.items():
    for n, fn in m.CANDIDATES.items():
        key = f"{a}/{n}"
        if a == "C3" and n in getattr(m, "PORTFOLIOS", set()):
            tr = m.run_portfolio(n, include_holdout=True)
        elif a == "C3" and n == "C3_ES_I2_40_15":
            tr = sim.run_signals(G["ES"], fn(G["ES"], LV["ES"])); tr["symbol"] = "ES"
        else:
            tr = sim.run_signals(g, fn(g, lv)); tr["symbol"] = "NQ"
        trades[key] = tr
        rows += row(key, tr, "primary" if key in prim_keys else "record")
D = load_d = None
sys.path.insert(0, os.path.join(HERE, "agents", "D"))
spec = importlib.util.spec_from_file_location("books", os.path.join(HERE, "agents", "D", "portfolios.py"))
books = importlib.util.module_from_spec(spec); spec.loader.exec_module(books)
for n, fn in books.BOOKS.items():
    tr = fn(include_holdout=True); trades["D/" + n] = tr
    rows += row("D/" + n, tr, "book")

summ = pd.DataFrame(rows)
cols = ["candidate", "kind", "split", "n", "win%", "avg", "total", "pf", "maxdd", "ret/dd", "sharpe_ann", "t", "worst_day", "worst_month"]
summ = summ[[c for c in cols if c in summ]]
summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)
for k, tr in trades.items():
    tr.to_csv(os.path.join(OUT, "trades", k.replace("/", "__") + ".csv"), index=False)

# ---- primary hypothesis tests (holdout only) ----
bh = base[base.split == "holdout"].set_index("date").pnl
tests = []
for code, (a, n) in PRIMARY.items():
    tr = trades[f"{a}/{n}"]; h = tr[tr.split == "holdout"]
    p = h.pnl.values
    t1 = st.ttest_1samp(p, 0, alternative="greater") if len(p) > 1 else None
    d = {"code": code, "candidate": f"{a}/{n}", "n": len(p), "avg": round(p.mean(), 1) if len(p) else np.nan,
         "p_abs": t1.pvalue if t1 else np.nan, "ci95": metrics.bootstrap_ci(p)}
    hd = h.groupby("date").pnl.sum()
    common = hd.index.intersection(bh.index)
    if len(common) > 1:
        diff = hd.loc[common] - bh.loc[common]
        tp = st.ttest_1samp(diff.values, 0, alternative="greater")
        d.update(n_paired=len(common), paired_diff=round(diff.mean(), 1), p_paired=tp.pvalue,
                 ci95_paired=metrics.bootstrap_ci(diff.values))
    tests.append(d)
T = pd.DataFrame(tests)
T["p_primary"] = np.where(T.code == "E1", T.get("p_paired"), T.p_abs)
order = T.p_primary.sort_values().index
m = len(T); holm = {}
prev = 0
for k, ix in enumerate(order):
    adj = min(1.0, max(prev, (m - k) * T.loc[ix, "p_primary"])); holm[ix] = adj; prev = adj
T["p_holm"] = pd.Series(holm)
T.to_csv(os.path.join(OUT, "primary_tests.csv"), index=False)
with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 30):
    print(T)
    print(summ[summ.split == "holdout"].to_string(index=False))
