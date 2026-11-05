import numpy as np
from a2lib import load, sim, disc_only
from ext import adr10
g, lv = load("NQ")
S = sim.slot_of; fam = "moc_grid"; adr = adr10(g)
def mk(sig="15:29", ref="pdc", exit_="15:59", thr=0.0, sl=None, sides=(1,-1)):
    out = []
    for r in lv.itertuples():
        i = r.i
        if not r.has_prev or not np.isfinite(adr[i]): continue
        c = g.C[i, S(sig)]; b = r.pdc if ref == "pdc" else r.open_0930
        if not (np.isfinite(c) and np.isfinite(b)) or c == b: continue
        m = c - b
        if abs(m) < thr * adr[i]: continue
        d = -int(np.sign(m))
        if d not in sides: continue
        out.append({"i": i, "entry_slot": S(sig)+1, "dir": d, "tp_pts": None, "sl_pts": sl, "exit_slot": S(exit_)})
    return out
tr = sim.run_signals(g, mk()); d = tr[tr.split=="discovery"]
print(d.groupby("dir").pnl.agg(["count","mean","sum"])); print(d.groupby("year").pnl.agg(["count","mean","sum"]))
d2 = d.copy(); d2["q"] = d2.date.str[:7]; print(d2.groupby(d2.date.str[:4] + "H" + ((d2.date.str[5:7].astype(int)>6)+1).astype(str)).pnl.agg(["count","mean"]))
for sig in ["15:14", "15:19", "15:24", "15:34", "15:39", "15:44"]:
    disc_only(g, sim.run_signals(g, mk(sig=sig)), f"sig={sig}", fam)
disc_only(g, sim.run_signals(g, mk(ref="open")), "ref=open0930", fam)
for sl in [20.0, 40.0]:
    disc_only(g, sim.run_signals(g, mk(sl=sl)), f"sl={sl}", fam)
disc_only(g, sim.run_signals(g, mk(exit_="15:49")), "exit 15:49", fam)
