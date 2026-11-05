"""Asia (18:00-02:00) and London (02:00-08:00) range sweeps in RTH, plus ON sweeps conditioned on prior-day value area."""
import numpy as np
from a2lib import load, sim, disc_only
from sweeps import sweep_signals
g, lv = load("NQ")
S = sim.slot_of
fam = "sess_sweep"
def rng_(a, b):
    out = {}
    for i in range(len(g)):
        h = g.H[i, S(a):S(b)]; l = g.L[i, S(a):S(b)]
        out[i] = (np.nanmax(h), np.nanmin(l)) if np.isfinite(h).any() else (np.nan, np.nan)
    return out
base = dict(tgt="R", R=2.0, last_idx=120, exit_idx=389, buf=2.0, stop_buf=5.0, max_risk=80.0)
for nm, (a, b) in {"asia": ("18:00", "02:00"), "london": ("02:00", "08:00"), "premkt": ("08:00", "09:30")}.items():
    ses = rng_(a, b)
    disc_only(g, sim.run_signals(g, sweep_signals(g, lv, level="sess", session=ses, **base)), f"{nm} sweep 2R", fam)
# ON sweep with value-area context: short only if swept ONH >= VAH (failed auction above value); long only if ONL <= VAL
f_va = lambda r, d, idx, c: (d == -1 and r.onh >= r.vah) or (d == 1 and r.onl <= r.val)
f_nva = lambda r, d, idx, c: not f_va(r, d, idx, c)
disc_only(g, sim.run_signals(g, sweep_signals(g, lv, level="on", extra_filter=f_va, **base)), "on sweep, level beyond VA", fam)
disc_only(g, sim.run_signals(g, sweep_signals(g, lv, level="on", extra_filter=f_nva, **base)), "on sweep, level inside VA", fam)
# prior-day VA edge sweep (open inside VA, poke beyond VAH/VAL, reclaim)
disc_only(g, sim.run_signals(g, sweep_signals(g, lv, level="va", **base)), "VA-edge sweep 2R", fam)
