from a2lib import load, sim, disc_only
from ext import ext_follow
g, lv = load("NQ")
fam = "ext_follow_grid"
base = dict(k=0.75, s=0.35, R=None, start=30, end=325, exit_idx=389)
tr = sim.run_signals(g, ext_follow(g, lv, **base)); disc_only(g, tr, "BASE", fam, log=False)
d = tr[tr.split=="discovery"]
print(d.groupby("dir").pnl.agg(["count","mean","sum"])); print(d.groupby("year").pnl.agg(["count","mean","sum"])); print(d.groupby("reason").pnl.agg(["count","mean","sum"]))
print(d.entry_time.str[:2].value_counts().sort_index())
grid = {"k": [0.5, 0.6, 0.9, 1.0], "s": [0.2, 0.25, 0.5, 0.7], "R": [1.5, 2.0, 3.0], "start": [0, 60, 90], "end": [205, 265, 355],
        "exit_idx": [329, 359], "n": [5, 20]}
for kk, vals in grid.items():
    for v in vals:
        kw = dict(base); kw[kk] = v
        disc_only(g, sim.run_signals(g, ext_follow(g, lv, **kw)), f"{kk}={v}", fam)
disc_only(g, sim.run_signals(g, ext_follow(g, lv, ref="pdc", **base)), "ref=pdc", fam)
