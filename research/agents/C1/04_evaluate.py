"""C1 step 4: evaluate the 3 FROZEN candidates (candidates.py, hash in FROZEN_candidates.sha256).
Discovery = selection period; validation 2025H2 = secondary check (already partly peeked at by Phase-1 leads).
Never loads 2026 (holdout locked). Writes out_04_*.csv and prints all report tables."""
import warnings; warnings.filterwarnings("ignore")
import os, sys, importlib.util
import numpy as np, pandas as pd
import c1lib as c
from lib import sim, strategies

g, lv = c.load()
assert not (g.split == "holdout").any()
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 50)

def ld(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp)
    sys.path.insert(0, os.path.dirname(path)); sp.loader.exec_module(m); return m
A3 = ld("a3cand", os.path.join(c.RES, "agents", "A3", "candidates.py"))
B1 = ld("b1cand", os.path.join(c.RES, "agents", "B1", "candidates.py"))
B2 = ld("b2cand", os.path.join(c.RES, "agents", "B2", "candidates.py"))
C1 = ld("c1cand", os.path.join(c.HERE, "candidates.py"))
CANDIDATES, A3X, HOLD, ORB_N = C1.CANDIDATES, C1.A3X, C1.HOLD, C1.ORB_N

STRATS = {
    "I2v1": lambda: strategies.i2_v1_signals(g, lv),
    "A3_C3": lambda: A3.c3_poc_sl60_np30(g, lv),
    "B1_I2_volpoor": lambda: B1.b1_i2_volpoor(g, lv),
    "B2_I2_TrendSize": lambda: B2.b2_i2_trendsize(g, lv),
}
for k, f in CANDIDATES.items():
    STRATS[k] = (lambda f=f: f(g, lv))
SIGS = {k: f() for k, f in STRATS.items()}
TR = {k: c.run(g, s) for k, s in SIGS.items()}
base = TR["I2v1"]
PER = [("discovery", *c.DISC), ("validation", *c.VAL)] + [(h, lo, hi) for h, lo, hi in c.HALVES]

# 1. main table
rows = []
for k, tr in TR.items():
    for pn, lo, hi in PER[:2]:
        s = c.stats(tr, g, lo, hi); s.update(strategy=k, period=pn); rows.append(s)
main = pd.DataFrame(rows)[["strategy", "period", "n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe", "worst", "lstreak"]]
main.to_csv("out_04_main.csv", index=False)
print("== 1. main table ==\n", main.to_string(index=False))

# 2. half-years
hy = pd.DataFrame({k: {h: f"{c.stats(tr, g, lo, hi)['n']} / {c.stats(tr, g, lo, hi)['avg']}" for h, lo, hi in c.HALVES} for k, tr in TR.items()}).T
print("\n== 2. half-years (n / avg) ==\n", hy.to_string())

# 3. paired vs I2 v1
rows = []
for k, tr in TR.items():
    if k == "I2v1":
        continue
    for pn, lo, hi in PER[:2]:
        a = tr[(tr.date >= lo) & (tr.date <= hi)]; b = base[(base.date >= lo) & (base.date <= hi)]
        p = c.paired_vs(a, b); dd = c.daily_diff_ci(tr, base, g, lo, hi)
        rows.append(dict(strategy=k, period=pn, common_days=p["n_common"], per_trade_diff=p["diff_avg"], ci=f"[{p['ci_lo']}, {p['ci_hi']}]",
                         daily_diff=dd["daily_diff"], daily_ci=f"[{dd['ci_lo']}, {dd['ci_hi']}]"))
pv = pd.DataFrame(rows); pv.to_csv("out_04_paired.csv", index=False)
print("\n== 3. paired vs I2 v1 (same days) and daily diff over all tradable days ==\n", pv.to_string(index=False))

# 4. I2-day vs non-I2-day
i2d = set(base.date)
rows = []
for k, tr in TR.items():
    for pn, lo, hi in PER[:2]:
        t = tr[(tr.date >= lo) & (tr.date <= hi)]
        on, off = t[t.date.isin(i2d)], t[~t.date.isin(i2d)]
        rows.append(dict(strategy=k, period=pn, on_n=len(on), on_avg=round(on.pnl.mean(), 0) if len(on) else None,
                         off_n=len(off), off_avg=round(off.pnl.mean(), 0) if len(off) else None))
print("\n== 4. on I2-v1 days vs other days ==\n", pd.DataFrame(rows).to_string(index=False))

# 5. execution realism
rows = []
for k in list(CANDIDATES) + ["A3_C3", "I2v1"]:
    s = SIGS[k]; tp = np.array([x["tp_pts"] for x in s if x.get("tp_pts") is not None])
    tt = c.run_through(g, s)
    tr = TR[k]
    sl1 = [dict(x, entry_slot=x["entry_slot"] + 1) for x in s]
    late = c.run(g, sl1)
    for pn, lo, hi in PER[:2]:
        m = np.array([lo <= g.dates[x["i"]] <= hi for x in s if x.get("tp_pts") is not None])
        a = c.stats(tr, g, lo, hi); b = c.stats(tt, g, lo, hi); l1 = c.stats(late, g, lo, hi)
        t2 = tr[(tr.date >= lo) & (tr.date <= hi)]
        rows.append(dict(strategy=k, period=pn, tgt_lt25=f"{np.mean(tp[m] < 25):.0%}", tgt_median=np.median(tp[m]),
                         base_avg=a["avg"], through_avg=b["avg"], through_sharpe=b["sharpe"], through_dd=b["maxdd"],
                         late1_avg=l1["avg"], late1_sharpe=l1["sharpe"],
                         cost_plus20_avg=round(a["avg"] - 20 * t2.contracts.mean(), 1)))
ex = pd.DataFrame(rows); ex.to_csv("out_04_execution.csv", index=False)
print("\n== 5. execution: targets<25, trade-through by 1 tick, entry 1 min late, +$20 cost ==\n", ex.to_string(index=False))

# 6. concentration
rows = []
for k, tr in TR.items():
    for pn, lo, hi in PER[:2]:
        p = np.sort(tr[(tr.date >= lo) & (tr.date <= hi)].pnl.values)[::-1]
        rows.append(dict(strategy=k, period=pn, avg=round(p.mean()), drop_best1=round(p[1:].mean()), drop_best5=round(p[5:].mean()),
                         drop_best10=round(p[10:].mean()) if len(p) > 10 else None, best=round(p[0]), median=round(np.median(p))))
print("\n== 6. concentration ==\n", pd.DataFrame(rows).to_string(index=False))

# 7. neighbours (each parameter one step up/down); validation shown for information only
def build(entry, ex, trend=True, volpoor=None):
    e = c.ent_i2(g, lv) if entry == "I2" else c.ent_orb(g, lv, entry, first=False)
    return c.overlay(c.make_signals(g, lv, e, ex), lv, volpoor=volpoor, trend=trend)
NB = {
    "C1_I2_A3X_TS": ("I2", A3X, [("poc_floor", 15), ("poc_floor", 25), ("sl", 45), ("sl", 75), ("np", (20, 20)), ("np", (45, 20)),
                                  ("np", (30, 15)), ("np", (30, 25)), ("exit", "1559")]),
    "C1_ORB15_POC_A3X_TS": (ORB_N, A3X, [("N", 10), ("N", 20), ("poc_floor", 15), ("poc_floor", 25), ("sl", 45), ("sl", 75),
                                         ("np", (20, 20)), ("np", (45, 20)), ("np", (30, 15)), ("np", (30, 25)), ("exit", "1559")]),
    "C1_ORB15_TP110_TS": (ORB_N, HOLD, [("N", 10), ("N", 20), ("tp", 90), ("tp", 130), ("sl", 120), ("sl", 170),
                                        ("exit", "15:30"), ("exit", "15:00"), ("exit", "1229")]),
}
rows = []
for k, (en, ex, steps) in NB.items():
    for p, v in [("base", None)] + steps:
        e2, x2 = en, dict(ex)
        if p == "N":
            e2 = v
        elif p != "base":
            x2[p] = v
        tr = c.run(g, build(e2, x2))
        d = c.stats(tr, g, *c.DISC); vv = c.stats(tr, g, *c.VAL)
        hp = sum((c.stats(tr, g, lo, hi)["avg"] or -1) > 0 for _, lo, hi in c.HALVES[:3])
        rows.append(dict(candidate=k, change=f"{p}={v}", D_n=d["n"], D_avg=d["avg"], D_dd=d["maxdd"], D_sharpe=d["sharpe"], D_halves_pos=hp,
                         V_n=vv["n"], V_avg=vv["avg"], V_dd=vv["maxdd"], V_sharpe=vv["sharpe"]))
nb = pd.DataFrame(rows); nb.to_csv("out_04_neighbors.csv", index=False)
print("\n== 7. neighbours ==\n", nb.to_string(index=False))
for k in NB:
    x = nb[(nb.candidate == k) & (nb.change != "base=None")]
    print(k, "disc positive", (x.D_avg > 0).sum(), "/", len(x), " disc>=50", (x.D_avg >= 50).sum(), " all-3-halves", (x.D_halves_pos == 3).sum(),
          " | val positive", (x.V_avg > 0).sum(), "/", len(x))

# 8. secondary variants (reported, not candidates): overlay ablations, volpoor stacking, hybrid, first-break ORB
rows = []
VAR = {
    "I2+A3X (no overlay)=A3_C3": build("I2", A3X, trend=False),
    "I2+A3X+TS (cand)": build("I2", A3X),
    "I2+A3X+volpoor": build("I2", A3X, trend=False, volpoor=0.25),
    "I2+A3X+TS+volpoor": build("I2", A3X, volpoor=0.25),
    "ORB15+A3X (no overlay)": build(15, A3X, trend=False),
    "ORB15+A3X+TS (cand)": build(15, A3X),
    "ORB15+A3X+TS+volpoor": build(15, A3X, volpoor=0.25),
    "ORB15+TP110 (no overlay)": build(15, HOLD, trend=False),
    "ORB15+TP110+TS (cand)": build(15, HOLD),
    "ORB15first+A3X+TS": c.overlay(c.make_signals(g, lv, c.ent_orb(g, lv, 15, True), A3X), lv, trend=True),
    "ORB15first+TP110+TS": c.overlay(c.make_signals(g, lv, c.ent_orb(g, lv, 15, True), HOLD), lv, trend=True),
    "DBL(I2 entry)+A3X+TS": c.overlay(c.make_signals(g, lv, c.ent_dbl(g, lv), A3X), lv, trend=True),
    "DBLlater+A3X+TS": c.overlay(c.make_signals(g, lv, c.ent_dbl(g, lv, True), A3X), lv, trend=True),
    "GAP0931+A3X+TS": c.overlay(c.make_signals(g, lv, c.ent_gap(g, lv), A3X), lv, trend=True),
    "I2_1x5+A3X+TS": c.overlay(c.make_signals(g, lv, c.ent_i2(g, lv, 1, 5), A3X), lv, trend=True),
}
i2e, o15 = c.ent_i2(g, lv), c.ent_orb(g, lv, 15, False)
first = {i: min(i2e.get(i, 10**9), o15.get(i, 10**9)) for i in set(i2e) | set(o15)}
VAR["FIRST(I2,ORB15)+A3X+TS"] = c.overlay(c.make_signals(g, lv, first, A3X), lv, trend=True)
for k, s in VAR.items():
    tr = c.run(g, s)
    for pn, lo, hi in PER[:2]:
        st = c.stats(tr, g, lo, hi)
        rows.append(dict(variant=k, period=pn, n=st["n"], avg=st["avg"], maxdd=st["maxdd"], retdd=st["ret/dd"], sharpe=st["sharpe"], worst=st["worst"], lstreak=st["lstreak"]))
sv = pd.DataFrame(rows); sv.to_csv("out_04_secondary.csv", index=False)
print("\n== 8. secondary variants ==\n", sv.to_string(index=False))

# 9. long/short split and monthly sign counts for candidates
rows = []
for k in CANDIDATES:
    tr = TR[k]
    for pn, lo, hi in PER[:2]:
        t = tr[(tr.date >= lo) & (tr.date <= hi)]
        mo = t.groupby(t.date.str[:7]).pnl.sum()
        rows.append(dict(strategy=k, period=pn, long_n=(t.dir == 1).sum(), long_avg=round(t[t.dir == 1].pnl.mean()), short_n=(t.dir == -1).sum(),
                         short_avg=round(t[t.dir == -1].pnl.mean()), months_pos=int((mo > 0).sum()), months_neg=int((mo <= 0).sum()),
                         half_size_share=f"{(t.contracts < 1).mean():.0%}", exit_reasons=t.reason.value_counts().to_dict()))
print("\n== 9. long/short, months, exits ==\n", pd.DataFrame(rows).to_string(index=False))
for k in CANDIDATES:
    TR[k].to_csv(f"trades_{k}.csv", index=False)
