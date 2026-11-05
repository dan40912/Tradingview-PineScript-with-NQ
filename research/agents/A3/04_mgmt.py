"""E5/E6: trade management on I2 v1 entries, DISCOVERY ONLY.
- 'no progress' exits (N minutes, X points), time stop relative to entry, absolute time stops
- partial exits (two half-size legs), runner with break-even / trailing stop
Base exit structures taken from the regions found in 02/03."""
import numpy as np, pandas as pd, itertools
from a3lib import load, i2_entries, with_exit, run, stats, noprog_exit_slot, RTH, EXIT_1229
from lib import sim

g, lv = load("NQ")
sigs = [s for s in i2_entries(g, lv) if g.split[s["i"]] == "discovery"]
K = ("n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
rows = []


def poc_tp(floor):
    def f(g_, r, s, px):
        return max(s["dir"] * (r.poc - px), floor)
    return f


BASES = {"55/145": (55, 145), "85/60": (85, 60), "100/60": (100, 60), "100/100": (100, 100), "120/60": (120, 60),
         "POCf20/60": (poc_tp(20), 60), "POCf20/100": (poc_tp(20), 100)}


def rec(name, base, tr, legs=1):
    # for multi-leg trades aggregate legs per date (one 'trade' = one day)
    if legs > 1:
        tr = tr.groupby(["date", "split", "year"], as_index=False).agg(pnl=("pnl", "sum"))
    st = stats(tr, g, "discovery")
    rows.append({"base": base, "rule": name, **{k: st.get(k) for k in K}})


for bname, (tp, sl) in BASES.items():
    rec("plain", bname, run(g, with_exit(g, lv, sigs, tp=tp, sl=sl)))
    # no-progress
    for N, X in itertools.product([20, 30, 45, 60, 90], [10, 20, 30]):
        xs = lambda g_, r, s, px, N=N, X=X: noprog_exit_slot(g_, s, px, N, X)
        rec(f"noprog N{N} X{X}", bname, run(g, with_exit(g, lv, sigs, tp=tp, sl=sl, exit_slot=xs)))
    # time stop relative to entry
    for M in [30, 45, 60, 90, 120]:
        xs = lambda g_, r, s, px, M=M: min(s["entry_slot"] + M, EXIT_1229)
        rec(f"tstop +{M}m", bname, run(g, with_exit(g, lv, sigs, tp=tp, sl=sl, exit_slot=xs)))
    for hh in ["10:30", "11:00", "11:30", "13:30", "15:55"]:
        rec(f"exit {hh}", bname, run(g, with_exit(g, lv, sigs, tp=tp, sl=sl, exit_slot=sim.slot_of(hh))))

# partial exits: 2 legs of 0.5 contract each (cost scales with contracts: 0.5*$48 per leg -> $48/day total)
for sl in [60, 100, 145]:
    for t1, t2 in itertools.product([30, 40, 55], [85, 100, 150, np.inf]):
        a = with_exit(g, lv, sigs, tp=t1, sl=sl, contracts=0.5)
        b = with_exit(g, lv, sigs, tp=t2, sl=sl, contracts=0.5)
        rec(f"partial T1={t1} T2={t2}", f"sl{sl}", run(g, a + b), legs=2)
        # runner moves stop to break-even (+0) once T1 reached
        b2 = with_exit(g, lv, sigs, tp=t2, sl=sl, contracts=0.5, be_trigger=t1, be_offset=0.0)
        rec(f"partial T1={t1} T2={t2} runnerBE", f"sl{sl}", run(g, a + b2), legs=2)
# trailing stops on full position
for sl in [60, 100, 145]:
    for trig, dist in itertools.product([40, 55, 85], [30, 50, 80]):
        rec(f"trail trig{trig} dist{dist} noTP", f"sl{sl}",
            run(g, with_exit(g, lv, sigs, tp=np.inf, sl=sl, trail_pts=dist, trail_trigger=trig)))
        rec(f"trail trig{trig} dist{dist} TP150", f"sl{sl}",
            run(g, with_exit(g, lv, sigs, tp=150, sl=sl, trail_pts=dist, trail_trigger=trig)))

df = pd.DataFrame(rows)
df.to_csv("out_mgmt.csv", index=False)
for b, grp in df.groupby("base", sort=False):
    print(f"\n===== base {b} =====")
    print(grp.drop(columns="base").to_string(index=False))
print("variants:", len(df))
