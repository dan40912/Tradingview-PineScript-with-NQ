import itertools, pandas as pd
from a2lib import load, sim, disc_only, summ
from sweeps import sweep_signals
g, lv = load("NQ")
fam = "on_sweep_grid"
base = dict(level="on", tgt="R", R=2.0, last_idx=120, exit_idx=389, buf=2.0, stop_buf=5.0, max_risk=80.0)
grid = {"R": [1.5, 2.5, 3.0], "last_idx": [90, 150], "exit_idx": [269, 329], "buf": [0.25, 5.0, 10.0], "stop_buf": [2.0, 10.0],
        "max_risk": [60.0, 100.0, 150.0], "min_risk": [5.0, 20.0]}
tr = sim.run_signals(g, sweep_signals(g, lv, **base)); disc_only(g, tr, "BASE", fam, log=False)
d = tr[tr.split=="discovery"]
print(d.groupby("dir").pnl.agg(["count","mean","sum"])); print(d.groupby("year").pnl.agg(["count","mean","sum"])); print(d.groupby("reason").pnl.agg(["count","mean","sum"]))
for k, vals in grid.items():
    for v in vals:
        kw = dict(base); kw[k] = v
        tr = sim.run_signals(g, sweep_signals(g, lv, **kw))
        disc_only(g, tr, f"{k}={v}", fam)
