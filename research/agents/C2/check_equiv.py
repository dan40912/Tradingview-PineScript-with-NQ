"""Check frozen candidates == exploration generators on discovery (no validation printed)."""
import numpy as np
import c2lib as L
from lib import sim
import sig
from candidates import CANDIDATES
g, lv, i2 = L.load()
im = sig.i2_entry_idx(g, lv)
a = sim.run_signals(g, CANDIDATES["C2_TE_OUTSIDE"](g, lv, sized=False))
b = sim.run_signals(g, sig.trend_ext(g, lv, day="outside", s=0.5, i2map=im))
print("TE equal:", len(a) == len(b) and np.allclose(a.pnl.values, b.pnl.values))
a = sim.run_signals(g, CANDIDATES["C2_PM_PULLBACK"](g, lv, sized=False))
b = sim.run_signals(g, sig.pm_pullback(g, lv, T=180, s=0.35))
print("PMP equal:", len(a) == len(b) and np.allclose(a.pnl.values, b.pnl.values))
for k, f in CANDIDATES.items():
    L.disc(g, sim.run_signals(g, f(g, lv)), f"FROZEN {k} sized", "frozen", log=False)
