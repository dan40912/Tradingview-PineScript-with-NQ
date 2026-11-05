"""Controls for the ON sweep idea: (a) same entries, flipped direction (same risk & 2R);
(b) random-direction / random-time placebo with the same structural stop rule (distance to the RTH extreme so far + 5, 10..80)."""
import numpy as np, pandas as pd
from a2lib import load, sim, disc_only, summ
from sweeps import sweep_signals
from lib.data import RTH_OFF
g, lv = load("NQ")
base = dict(level="on", tgt="R", R=2.0, last_idx=120, exit_idx=389, buf=2.0, stop_buf=5.0, max_risk=80.0)
sg = sweep_signals(g, lv, **base)
fl = [dict(s, dir=-s["dir"]) for s in sg]
disc_only(g, sim.run_signals(g, fl), "flipped dir", "on_sweep_ctrl")
# placebo: for each discovery day with has_prev, random 5m decision bar in [4,120], random dir; stop = dist to extreme against + 5
rng = np.random.default_rng(1)
res = []
for rep in range(20):
    sigs = []
    for r in lv[lv.has_prev & (lv.split == "discovery")].itertuples():
        if rng.random() > 0.5: continue
        i = r.i; idx = int(rng.choice(np.arange(4, 121, 5)))
        if 28 <= idx+1 <= 32: continue
        H = g.H[i, RTH_OFF:RTH_OFF+idx+1]; L = g.L[i, RTH_OFF:RTH_OFF+idx+1]; c = g.C[i, RTH_OFF+idx]
        if not np.isfinite(c): continue
        d = int(rng.choice([1, -1]))
        ext = np.nanmax(H) if d == -1 else np.nanmin(L)
        risk = max(abs(ext-c)+5, 10)
        if risk > 80: continue
        sigs.append({"i": i, "entry_slot": RTH_OFF+idx+1, "dir": d, "tp_pts": 2*risk, "sl_pts": risk, "exit_slot": RTH_OFF+389})
    tr = sim.run_signals(g, sigs)
    res.append(tr.pnl.mean())
print("placebo avg per trade over 20 reps: mean %.1f sd %.1f max %.1f" % (np.mean(res), np.std(res), np.max(res)))
