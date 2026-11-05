"""Neighbourhood robustness (discovery only) around the three pre-candidate structures:
 C1 fixed TP/SL; C2 fixed TP/SL + no-progress exit; C3 POC target (with floor) / SL + no-progress exit."""
import numpy as np, pandas as pd, itertools
from a3lib import load, i2_entries, with_exit, run, stats, noprog_exit_slot

g, lv = load("NQ")
sigs = [s for s in i2_entries(g, lv) if g.split[s["i"]] == "discovery"]
K = ("n", "win%", "avg", "maxdd", "ret/dd", "sharpe_ann")
rows = []


def poc_tp(floor):
    return lambda g_, r, s, px: max(s["dir"] * (r.poc - px), floor)


for tp, sl in itertools.product([85, 100, 120], [45, 60, 80]):
    st = stats(run(g, with_exit(g, lv, sigs, tp=tp, sl=sl)), g)
    rows.append({"fam": "C1", "tp": tp, "sl": sl, "N": None, "X": None, **{k: st[k] for k in K}})
for tp, sl, N, X in itertools.product([85, 100, 120], [45, 60, 80], [20, 30, 45], [10, 20, 30]):
    xs = lambda g_, r, s, px, N=N, X=X: noprog_exit_slot(g_, s, px, N, X)
    st = stats(run(g, with_exit(g, lv, sigs, tp=tp, sl=sl, exit_slot=xs)), g)
    rows.append({"fam": "C2", "tp": tp, "sl": sl, "N": N, "X": X, **{k: st[k] for k in K}})
for fl, sl, N, X in itertools.product([10, 20, 30], [45, 60, 80], [20, 30, 45], [10, 20, 30]):
    xs = lambda g_, r, s, px, N=N, X=X: noprog_exit_slot(g_, s, px, N, X)
    st = stats(run(g, with_exit(g, lv, sigs, tp=poc_tp(fl), sl=sl, exit_slot=xs)), g)
    rows.append({"fam": "C3", "tp": f"POCf{fl}", "sl": sl, "N": N, "X": X, **{k: st[k] for k in K}})
df = pd.DataFrame(rows)
df.to_csv("out_neighbors.csv", index=False)
pd.set_option("display.width", 200)
for f, grp in df.groupby("fam"):
    print(f"\n== {f}: {len(grp)} cells, positive avg {int((grp.avg > 0).sum())}, avg>=50: {int((grp.avg >= 50).sum())}")
    print(grp[list(K[2:])].astype(float).describe().loc[["min", "25%", "50%", "75%", "max"]].round(2))
print(df[df.fam == "C1"].to_string(index=False))
c = df[(df.fam == "C2") & (df.tp == 100) & (df.sl == 60)]
print(c.to_string(index=False))
c = df[(df.fam == "C3") & (df.tp == "POCf20") & (df.sl == 60)]
print(c.to_string(index=False))
print("variants:", len(df))
