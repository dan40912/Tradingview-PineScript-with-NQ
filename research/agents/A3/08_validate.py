"""Validation of the PRE-REGISTERED A3 candidates (frozen in candidates.py before this script was run).
Pre-registered list (nothing else is evaluated on validation):
  baseline I2 v1 (55/145), A3_C1, A3_C2, A3_C3              -> main table
  neighbour cells of each candidate (each param +-1 grid step) -> report share positive (no re-selection)
  secondary generalization checks (not candidates): C2 exits on I2+H1 entries; ES I2 prior 22/58 vs ES 45/20
Never touches 2026 (load_grid default hides holdout)."""
import numpy as np, pandas as pd, itertools
from a3lib import load, i2_entries, with_exit, run, stats, noprog_exit_slot, ndays
from lib import metrics, strategies, sim
from candidates import ALL_PREREGISTERED as CANDIDATES

g, lv = load("NQ")
assert not (g.split == "holdout").any()
K = ("n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann")
pd.set_option("display.width", 250)
rows, trades = [], {}
base = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
trades["I2v1 base 55/145"] = base
for nm, fn in CANDIDATES.items():
    trades[nm] = sim.run_signals(g, fn(g, lv))
for nm, tr in trades.items():
    for sp in ["discovery", "validation", "ALL"]:
        st = stats(tr, g, sp)
        rows.append({"strategy": nm, "split": sp, **{k: st.get(k) for k in K}, "t": st.get("t")})
main = pd.DataFrame(rows)
main.to_csv("out_validation_main.csv", index=False)
print(main.to_string(index=False))

# by year / reason
for nm, tr in trades.items():
    print(f"\n-- {nm}: by year");
    print(metrics.by_period(tr, "year")[["n", "win%", "avg", "total", "maxdd"]])
    v = tr[tr.split == "validation"]
    print("validation exit reasons:", v.groupby("reason").pnl.agg(["count", "mean", "sum"]).round(0).to_dict("index"))

# paired difference vs baseline (same days)
print("\n== paired per-trade difference vs I2 v1 base (same dates) ==")
for nm, tr in trades.items():
    if nm.startswith("I2v1"):
        continue
    m = base[["date", "split", "pnl"]].merge(tr[["date", "pnl"]], on="date", suffixes=("_b", "_c"))
    for sp in ["discovery", "validation"]:
        d = (m[m.split == sp].pnl_c - m[m.split == sp].pnl_b).values
        print(f"{nm:28s} {sp:10s} n={len(d)} mean diff={d.mean():7.1f}  95%CI={metrics.bootstrap_ci(d)}  share better={np.mean(d > 0):.2f} worse={np.mean(d < 0):.2f}")

# overlap with I2 trading days
print("\n== overlap: candidate trades on I2-v1 days vs other days ==")
bd = set(base.date)
for nm, tr in trades.items():
    on = tr.date.isin(bd)
    print(f"{nm:28s} on I2 days: {on.sum()}  on non-I2 days: {(~on).sum()}")

# neighbour cells on validation (report only)
def poc_tp(fl):
    return lambda g_, r, s, px: max(s["dir"] * (r.poc - px), fl)

sigV = [s for s in i2_entries(g, lv) if g.split[s["i"]] == "validation"]
nb = []
for tp, sl in itertools.product([85, 100, 120], [45, 60, 80]):
    st = stats(run(g, with_exit(g, lv, sigV, tp=tp, sl=sl)), g, "validation")
    nb.append({"fam": "C1", "p": f"{tp}/{sl}", "avg": st["avg"], "sharpe": st["sharpe_ann"], "maxdd": st["maxdd"]})
for tp, sl, N, X in itertools.product([85, 100, 120], [45, 60, 80], [20, 30, 45], [10, 20, 30]):
    xs = lambda g_, r, s, px, N=N, X=X: noprog_exit_slot(g_, s, px, N, X)
    st = stats(run(g, with_exit(g, lv, sigV, tp=tp, sl=sl, exit_slot=xs)), g, "validation")
    nb.append({"fam": "C2", "p": f"{tp}/{sl}/N{N}/X{X}", "avg": st["avg"], "sharpe": st["sharpe_ann"], "maxdd": st["maxdd"]})
for fl, sl, N, X in itertools.product([10, 20, 30], [45, 60, 80], [20, 30, 45], [10, 20, 30]):
    xs = lambda g_, r, s, px, N=N, X=X: noprog_exit_slot(g_, s, px, N, X)
    st = stats(run(g, with_exit(g, lv, sigV, tp=poc_tp(fl), sl=sl, exit_slot=xs)), g, "validation")
    nb.append({"fam": "C3", "p": f"f{fl}/{sl}/N{N}/X{X}", "avg": st["avg"], "sharpe": st["sharpe_ann"], "maxdd": st["maxdd"]})
nb = pd.DataFrame(nb)
nb.to_csv("out_validation_neighbors.csv", index=False)
print("\n== validation neighbourhood (report only; base I2 v1 validation avg =",
      stats(base, g, "validation")["avg"], ") ==")
for f, grp in nb.groupby("fam"):
    print(f, len(grp), "cells; avg>0:", int((grp.avg > 0).sum()), "; avg > base:", int((grp.avg > stats(base, g, 'validation')['avg']).sum()),
          "; median avg", grp.avg.median(), "; median maxdd", grp.maxdd.median())
print(nb[nb.fam == "C1"].to_string(index=False))

# secondary checks
print("\n== secondary: C2 exits on I2+H1 entries ==")
sigH = i2_entries(g, lv, h1=True)
xs = lambda g_, r, s, px: noprog_exit_slot(g_, s, px, 30, 20)
for lab, kw in [("H1 55/145", dict(tp=55, sl=145)), ("H1 C2", dict(tp=100, sl=60, exit_slot=xs))]:
    tr = run(g, with_exit(g, lv, sigH, **kw))
    for sp in ["discovery", "validation"]:
        print(lab, sp, {k: stats(tr, g, sp).get(k) for k in K})
print("\n== secondary: ES I2 ==")
gE, lvE = load("ES")
sigE = i2_entries(gE, lvE)
for lab, kw in [("ES 22/58 (prior)", dict(tp=22, sl=58)), ("ES 45/20 (flipped)", dict(tp=45, sl=20))]:
    tr = run(gE, with_exit(gE, lvE, sigE, **kw))
    for sp in ["discovery", "validation"]:
        print(lab, sp, {k: stats(tr, gE, sp).get(k) for k in K})
print("validation-stage evaluations:", len(trades) + len(nb) + 4)
