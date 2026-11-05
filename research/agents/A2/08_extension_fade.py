"""Excess-extension reversion: first 5m close (10:00..14:55) where |close - 09:30 open| > k * ADR10 -> fade.
ADR10 = mean of the previous 10 RTH high-low ranges (strictly prior days). Stop = s*ADR10, target = R * stop or none; exit 15:59."""
import itertools, numpy as np
from a2lib import load, sim, disc_only
from lib.data import RTH_OFF
g, lv = load("NQ")
rH, rL = g.rth(g.H), g.rth(g.L)
rng = np.nanmax(rH, 1) - np.nanmin(rL, 1)
adr = np.full(len(g), np.nan)
idxs = np.where(g.has_rth)[0]
for n, i in enumerate(idxs):
    if n >= 10:
        adr[i] = np.mean(rng[idxs[n-10:n]])
def sigs(k, s, R, start=30, end=325, fade=True):
    out = []
    for r in lv.itertuples():
        i = r.i
        if not r.has_prev or not np.isfinite(adr[i]) or not np.isfinite(r.open_0930): continue
        C = g.C[i, RTH_OFF:RTH_OFF+390]
        for idx in range(start + 4, end, 5):
            c = C[idx]
            if not np.isfinite(c): continue
            m = c - r.open_0930
            if abs(m) > k * adr[i]:
                d = -int(np.sign(m)) if fade else int(np.sign(m))
                out.append({"i": i, "entry_slot": RTH_OFF+idx+1, "dir": d, "sl_pts": s*adr[i],
                            "tp_pts": (R*s*adr[i] if R else None), "exit_slot": RTH_OFF+389})
                break
    return out
for k, s, R in itertools.product([0.5, 0.75, 1.0], [0.2, 0.35], [1.0, 2.0, None]):
    disc_only(g, sim.run_signals(g, sigs(k, s, R)), f"fade k={k} s={s} R={R}", "ext_fade")
for k in [0.5, 0.75]:
    disc_only(g, sim.run_signals(g, sigs(k, 0.35, None, fade=False)), f"FOLLOW k={k} s=0.35 R=None", "ext_fade")
