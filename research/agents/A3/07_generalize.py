"""E9: do the exit rules generalize? DISCOVERY ONLY.
 (a) NQ I2 by sub-period (2024 / 2025H1), long vs short, entry-time bucket
 (b) NQ I2 + H1 filter entries
 (c) ES I2 entries (points scaled by the discovery median ATR14 ratio ES/NQ)"""
import numpy as np, pandas as pd
from a3lib import load, i2_entries, with_exit, run, stats, noprog_exit_slot

K = ("n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann")
pd.set_option("display.width", 220)
rows = []


def poc_tp(floor):
    return lambda g_, r, s, px: max(s["dir"] * (r.poc - px), floor)


def NP(N, X):
    return lambda g_, r, s, px: noprog_exit_slot(g_, s, px, N, X)


def structs(k=1.0):
    return {"base 55/145": dict(tp=55 * k, sl=145 * k),
            "C1 100/60": dict(tp=100 * k, sl=60 * k),
            "C2 100/60+NP30/20": dict(tp=100 * k, sl=60 * k, exit_slot=NP(30, 20 * k)),
            "C3 POCf20/60+NP30/20": dict(tp=poc_tp(20 * k), sl=60 * k, exit_slot=NP(30, 20 * k))}


gN, lvN = load("NQ")
sigN = [s for s in i2_entries(gN, lvN) if gN.split[s["i"]] == "discovery"]
sigH1 = [s for s in i2_entries(gN, lvN, h1=True) if gN.split[s["i"]] == "discovery"]
for sn, kw in structs().items():
    tr = run(gN, with_exit(gN, lvN, sigN, **kw))
    st = stats(tr, gN)
    rows.append({"set": "NQ I2", "struct": sn, **{k: st[k] for k in K}})
    for y in ["2024", "2025"]:
        t = tr[tr.year == y]
        st = stats(t, gN)  # sharpe here uses all discovery days -> only compare n/avg/maxdd
        rows.append({"set": f"NQ I2 {y}(disc)", "struct": sn, **{k: st[k] for k in K}})
    for d, nm in [(1, "long"), (-1, "short")]:
        st = stats(tr[tr.dir == d], gN)
        rows.append({"set": f"NQ I2 {nm}", "struct": sn, **{k: st[k] for k in K}})
    early = tr.entry_slot < 930 + 30
    st = stats(tr[early], gN); rows.append({"set": "NQ I2 entry<10:00", "struct": sn, **{k: st[k] for k in K}})
    st = stats(tr[~early], gN); rows.append({"set": "NQ I2 entry>=10:00", "struct": sn, **{k: st[k] for k in K}})
    tr = run(gN, with_exit(gN, lvN, sigH1, **kw))
    st = stats(tr, gN)
    rows.append({"set": "NQ I2+H1", "struct": sn, **{k: st[k] for k in K}})

gE, lvE = load("ES")
sigE = [s for s in i2_entries(gE, lvE) if gE.split[s["i"]] == "discovery"]
dN = lvN[lvN.split == "discovery"].atr14.median(); dE = lvE[lvE.split == "discovery"].atr14.median()
k = dE / dN
print(f"ATR14 median NQ {dN:.1f} ES {dE:.1f} -> scale {k:.3f}")
for sn, kw in structs(k).items():
    tr = run(gE, with_exit(gE, lvE, sigE, **kw))
    st = stats(tr, gE)
    rows.append({"set": f"ES I2 (x{k:.2f})", "struct": sn, **{kk: st[kk] for kk in K}})
tr = run(gE, with_exit(gE, lvE, sigE, tp=22, sl=58))
st = stats(tr, gE); rows.append({"set": "ES I2", "struct": "prior ES 22/58", **{kk: st[kk] for kk in K}})
# ES fixed-point small grid for context (rows tp, cols sl), discovery
grid = []
for tp in [10, 15, 22, 30, 40, 55]:
    for sl in [10, 15, 22, 30, 45, 58]:
        st = stats(run(gE, with_exit(gE, lvE, sigE, tp=tp, sl=sl)), gE)
        grid.append({"tp": tp, "sl": sl, "avg": st["avg"], "sharpe": st["sharpe_ann"]})
df = pd.DataFrame(rows)
df.to_csv("out_generalize.csv", index=False)
print(df.to_string(index=False))
gd = pd.DataFrame(grid)
print("\nES I2 discovery fixed-point grid avg $/trade (rows TP, cols SL):")
print(gd.pivot(index="tp", columns="sl", values="avg"))
print(gd.pivot(index="tp", columns="sl", values="sharpe"))
print("variants:", len(df) + len(gd))
