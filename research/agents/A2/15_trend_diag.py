"""Diagnostics for the only surviving candidate (no parameter changes): neighbor grid in BOTH periods (information only),
monthly P&L, bootstrap CI, entry-time distribution, early-close/halt handling."""
import numpy as np, pandas as pd
from a2lib import load, sim, summ, metrics
from ext import ext_follow
from candidates import trend_ext_follow
g, lv = load("NQ")
tr = sim.run_signals(g, trend_ext_follow(g, lv))
rows = []
base = dict(k=0.75, s=0.5, n=10, start=30, end=325, exit_idx=359)
grid = {"k": [0.6, 0.9], "s": [0.35, 0.7], "n": [5, 15], "start": [0, 60], "end": [265, 355], "exit_idx": [329, 389]}
for kk, vals in [("BASE", [None])] + list(grid.items()):
    for v in vals:
        kw = dict(base)
        if kk != "BASE": kw[kk] = v
        t = sim.run_signals(g, ext_follow(g, lv, **kw))
        d, va = summ(g, t, "discovery"), summ(g, t, "validation")
        rows.append({"param": kk, "val": v, "D_n": d["n"], "D_avg": d["avg"], "D_sh": d["sharpe_ann"], "V_n": va["n"], "V_avg": va["avg"], "V_sh": va["sharpe_ann"]})
nb = pd.DataFrame(rows); print(nb.to_string(index=False)); nb.to_csv("results_trend_neighbors.csv", index=False)
pd.DataFrame([{"family": "trend_diag_neighbors", "name": f"{r.param}={r.val}", "d_n": r.D_n, "d_avg": r.D_avg} for r in nb.itertuples()]).to_csv("variant_log.csv", mode="a", header=False, index=False)
tr["m"] = tr.date.str[:7]
print(tr.groupby("m").pnl.agg(["count", "sum"]).T.to_string())
for sp in ["discovery", "validation"]:
    x = tr[tr.split == sp]
    print(sp, "bootstrap 95% CI of avg:", metrics.bootstrap_ci(x.pnl.values), "t=", round(x.pnl.mean() / x.pnl.std() * np.sqrt(len(x)), 2))
print(tr.entry_time.str[:2].value_counts().sort_index().to_dict())
print(tr.groupby(["split", "dir"]).pnl.agg(["count", "mean"]).round(0))
