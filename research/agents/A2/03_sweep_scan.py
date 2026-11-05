import sys, itertools
from a2lib import load, sim, disc_only
from sweeps import sweep_signals
g, lv = load("NQ")
fam = "sweep_scan1"
for level, tgt, R, last, ex in itertools.product(["on", "pd"], ["R", "mid", "none"], [1.0, 2.0], [60, 120], [179, 389]):
    if tgt != "R" and R != 1.0: continue
    sg = sweep_signals(g, lv, level=level, tgt=tgt, R=R, last_idx=last, exit_idx=ex)
    tr = sim.run_signals(g, sg)
    disc_only(g, tr, f"{level} tgt={tgt}{R if tgt=='R' else ''} last={last} ex={ex}", fam)
