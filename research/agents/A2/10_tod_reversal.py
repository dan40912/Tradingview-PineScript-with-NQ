"""Time-of-day reversal ideas from screen 02 (each is a multiple-testing survivor, so treat with suspicion):
(a) 15:30 entry, direction = -sign(15:29 close - prior RTH close), exit 15:59 close (MOC reversal)
(b) 10:00 entry, direction = -sign(09:59 close - prior close), exit 11:29 (10 o'clock reversal)
(c) 15:30 entry, direction = -sign(14:59 - 13:59 move) exit 15:59
optional stop; optional min-move threshold in ADR units."""
import numpy as np
from a2lib import load, sim, disc_only
from ext import adr10
g, lv = load("NQ")
S = sim.slot_of; fam = "tod_rev"
adr = adr10(g)
def mk(sig_slot, ref, entry, exit_, thr=0.0, sl=None):
    out = []
    for r in lv.itertuples():
        i = r.i
        if not r.has_prev or not np.isfinite(adr[i]): continue
        c = g.C[i, sig_slot]
        b = r.pdc if ref == "pdc" else g.C[i, ref]
        if not (np.isfinite(c) and np.isfinite(b)): continue
        m = c - b
        if abs(m) < thr * adr[i] or m == 0: continue
        out.append({"i": i, "entry_slot": entry, "dir": -int(np.sign(m)), "tp_pts": None, "sl_pts": sl, "exit_slot": exit_})
    return out
for thr in [0.0, 0.25, 0.5]:
    disc_only(g, sim.run_signals(g, mk(S("15:29"), "pdc", S("15:30"), S("15:59"), thr)), f"(a) MOC rev thr={thr}", fam)
    disc_only(g, sim.run_signals(g, mk(S("09:59"), "pdc", S("10:00"), S("11:29"), thr)), f"(b) 10:00 rev thr={thr}", fam)
    disc_only(g, sim.run_signals(g, mk(S("14:59"), S("13:59"), S("15:30"), S("15:59"), thr/2)), f"(c) 14-15 rev thr={thr/2}", fam)
