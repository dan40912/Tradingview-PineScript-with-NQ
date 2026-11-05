"""Post-validation diagnostics for the surviving candidate A3_C3 (NO parameter changes; descriptive only).
 - execution stress: target needs 0.5-pt trade-through, extra slippage, 1-2 min late entry
 - ablation (which part does the work) - post hoc, descriptive
 - long/short, entry-time buckets, worst day/month, exit reasons
 - ES and I2+H1 with the same rule (post hoc generalization check)"""
import numpy as np, pandas as pd
from a3lib import load, i2_entries, with_exit, run, stats, noprog_exit_slot
from lib import sim, metrics, strategies
from candidates import CANDIDATES, POC_FLOOR

K = ("n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann")
pd.set_option("display.width", 250)
g, lv = load("NQ")
fn = CANDIDATES["A3_C3_POC_SL60_NP30"]
rows = []


def rec(label, tr, gg=g):
    for sp in ["discovery", "validation"]:
        st = stats(tr, gg, sp)
        rows.append({"test": label, "split": sp, **{k: st.get(k) for k in K}})


c3 = sim.run_signals(g, fn(g, lv))
rec("C3 as frozen", c3)
# trade-through: target must be exceeded by 0.5 pt, but filled at the target price
sg = fn(g, lv)
tt = sim.run_signals(g, [dict(s, tp_pts=s["tp_pts"] + 0.5) for s in sg])
tt.loc[tt.reason == "tp", "pnl"] -= 0.5 * g.point
rec("target needs +0.5pt trade-through", tt)
for extra in [10, 20, 40]:
    t = c3.copy(); t["pnl"] -= extra
    rec(f"extra slippage ${extra}/trade", t)
# late entry: shift the I2 entry by k minutes (target/NP recomputed from the new fill)
for k in [1, 2]:
    base_sigs = [dict(s, entry_slot=s["entry_slot"] + k) for s in strategies.i2_v1_signals(g, lv)]
    tp = lambda g_, r, s, px: max(s["dir"] * (r.poc - px), POC_FLOOR)
    xs = lambda g_, r, s, px: noprog_exit_slot(g_, s, px, 30, 20)
    rec(f"entry {k} min late", run(g, with_exit(g, lv, base_sigs, tp=tp, sl=60, exit_slot=xs)))
# ablation
sigs = i2_entries(g, lv)
tp = lambda g_, r, s, px: max(s["dir"] * (r.poc - px), POC_FLOOR)
xs = lambda g_, r, s, px: noprog_exit_slot(g_, s, px, 30, 20)
rec("ablation: POC target, SL145, no NP", run(g, with_exit(g, lv, sigs, tp=tp, sl=145)))
rec("ablation: POC target, SL60, no NP", run(g, with_exit(g, lv, sigs, tp=tp, sl=60)))
rec("ablation: POC target, SL145, NP30/20", run(g, with_exit(g, lv, sigs, tp=tp, sl=145, exit_slot=xs)))
rec("ablation: TP55, SL60, NP30/20", run(g, with_exit(g, lv, sigs, tp=55, sl=60, exit_slot=xs)))
rec("ablation: TP55, SL145, NP30/20", run(g, with_exit(g, lv, sigs, tp=55, sl=145, exit_slot=xs)))
rec("ablation: TP30, SL60, NP30/20", run(g, with_exit(g, lv, sigs, tp=30, sl=60, exit_slot=xs)))
# generalization
sigH = i2_entries(g, lv, h1=True)
rec("I2+H1 entries with C3 exits", run(g, with_exit(g, lv, sigH, tp=tp, sl=60, exit_slot=xs)))
gE, lvE = load("ES")
k = 0.2  # ES/NQ ATR14 ratio measured on discovery (07_generalize.py: 0.199)
tpE = lambda g_, r, s, px: max(s["dir"] * (r.poc - px), POC_FLOOR * k)
xsE = lambda g_, r, s, px: noprog_exit_slot(g_, s, px, 30, 20 * k)
rec("ES I2 with C3 exits (scaled x0.2)", run(gE, with_exit(gE, lvE, i2_entries(gE, lvE), tp=tpE, sl=60 * k, exit_slot=xsE)), gE)
rec("ES I2 with C3 exits (SL 12, unscaled POC, NP 4pt)", run(gE, with_exit(gE, lvE, i2_entries(gE, lvE), tp=lambda g_, r, s, px: max(s["dir"] * (r.poc - px), 4), sl=24, exit_slot=xsE)), gE)
df = pd.DataFrame(rows)
df.to_csv("out_c3_stress.csv", index=False)
print(df.to_string(index=False))

print("\n-- C3 details --")
c3["tp_target"] = [s["tp_pts"] for s in sg]
for sp in ["discovery", "validation"]:
    t = c3[c3.split == sp]
    print(sp, "target pts quantiles", t.tp_target.quantile([.1, .25, .5, .75, .9]).round(0).to_dict(),
          "| floor used:", int((t.tp_target <= POC_FLOOR + 1e-9).sum()))
    print("  reasons", t.groupby("reason").pnl.agg(["count", "mean", "sum"]).round(0).to_dict("index"))
    print("  long", stats(t[t.dir == 1], g, sp)["avg"], "n", (t.dir == 1).sum(), "| short", stats(t[t.dir == -1], g, sp)["avg"], "n", (t.dir == -1).sum())
    e = t.entry_slot < 960
    print("  entry<10:00 avg", round(t[e].pnl.mean(), 1), "n", e.sum(), "| >=10:00 avg", round(t[~e].pnl.mean(), 1), "n", (~e).sum())
    print("  worst trade", t.pnl.min(), "best", t.pnl.max(), "| median holding min", (t.exit_slot - t.entry_slot).median())
    mo = t.groupby(t.date.str[:7]).pnl.sum()
    print("  months +/-:", int((mo > 0).sum()), "/", int((mo <= 0).sum()), "worst month", mo.min())
b = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
print("\nbaseline worst trade", b.pnl.min(), "; C3 worst trade", c3.pnl.min())
print("max consecutive losses base / C3:",
      max((len(list(gr)) for k_, gr in __import__("itertools").groupby(b.pnl < 0) if k_), default=0),
      max((len(list(gr)) for k_, gr in __import__("itertools").groupby(c3.pnl < 0) if k_), default=0))
