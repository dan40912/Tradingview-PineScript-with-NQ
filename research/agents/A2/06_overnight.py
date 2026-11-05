"""Night effect. (1) Engine trades fully inside one grid day: long 18:00 -> exit at various overnight slots.
(2) Separate close-to-open calc (cannot be done in engine: spans two grid days): long at prior RTH close (15:59 close)
-> next day's 09:30 open; and open-to-close (09:30 open -> 15:59 close) in engine. Costs $48 each trade."""
import numpy as np, pandas as pd
from a2lib import load, sim, disc_only, summ, metrics, n_tradable
g, lv = load("NQ")
S = sim.slot_of
fam = "overnight"
ok = lv.tradable.values & lv.has_prev.values
for ex in ["02:59", "08:29", "09:29"]:
    for sl in [None, 100.0]:
        sigs = [{"i": i, "entry_slot": 0, "dir": 1, "tp_pts": None, "sl_pts": sl, "exit_slot": S(ex)} for i in np.where(ok)[0]]
        disc_only(g, sim.run_signals(g, sigs), f"long 18:00->{ex} sl={sl}", fam)
sigs = [{"i": i, "entry_slot": 930, "dir": 1, "tp_pts": None, "sl_pts": None, "exit_slot": 1319} for i in np.where(ok)[0]]
disc_only(g, sim.run_signals(g, sigs), "long 09:30->15:59 (open-to-close)", fam)
# close-to-open, own calc
rows = []
for i in np.where(ok)[0]:
    j = g.prev_rth_day(i)
    pc = lv.pdc[i]; o = lv.open_0930[i]
    if np.isfinite(pc) and np.isfinite(o):
        rows.append({"date": g.dates[i], "pts": o - pc, "pnl": (o - pc) * g.point - g.cost, "split": g.split[i], "year": g.year[i]})
co = pd.DataFrame(rows)
print("close->open (own calc), discovery:", metrics.summary(co[co.split == "discovery"], n_days=n_tradable(g, "discovery")))
pd.DataFrame([{"family": fam, "name": "close->open own calc", "d_avg": co[co.split=="discovery"].pnl.mean()}]).to_csv("variant_log.csv", mode="a", header=False, index=False) if False else None
