"""B3: evaluate frozen candidates (discovery / validation only), neighbours, I2 overlap, long/short."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from lib import data, features, sim, metrics, strategies
import candidates as C

pd.set_option("display.width", 250)
g = data.load_grid("NQ"); lv = features.daily_levels(g); es = data.load_grid("ES")
assert g.dates[-1] < "2026-01-01"
nd = {s: int(((g.split == s) & g.tradable).sum()) for s in ["discovery", "validation"]}
i2 = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
i2_days = set(i2.date)


def split_table(tr):
    out = {}
    for s in ["discovery", "validation"]:
        out[s] = metrics.summary(tr[tr.split == s], n_days=nd[s])
    return pd.DataFrame(out).T


def avg_by_split(tr):
    return {s: round(tr[tr.split == s].pnl.mean(), 1) if (tr.split == s).any() else np.nan for s in ["discovery", "validation"]}


rows_out = []
for name, fn in {**C.CANDIDATES, **C.REJECTED}.items():
    tr = sim.run_signals(g, fn(g, lv))
    print("=" * 30, name)
    print(split_table(tr).to_string())
    print(metrics.by_period(tr, "year").to_string())
    tr["half"] = np.where(tr.date < "2025-01-01", "2024", np.where(tr.date <= "2025-06-30", "2025H1", "2025H2"))
    print(tr.groupby("half").pnl.agg(["count", "mean", "sum"]).round(1).to_string())
    print("long/short:"); print(tr.groupby(["split", "dir"]).pnl.agg(["count", "mean", "sum"]).round(1).to_string())
    print("exit reasons:"); print(tr.groupby(["split", "reason"]).pnl.agg(["count", "mean"]).round(1).to_string())
    tr["i2day"] = tr.date.isin(i2_days)
    print("I2 overlap:"); print(tr.groupby(["split", "i2day"]).pnl.agg(["count", "mean", "sum"]).round(1).to_string())
    print("bootstrap 95% CI avg disc:", metrics.bootstrap_ci(tr[tr.split == "discovery"].pnl), " val:", metrics.bootstrap_ci(tr[tr.split == "validation"].pnl))
    # combined with I2 (one contract each, independent)
    comb = pd.concat([tr[["date", "pnl", "split", "year"]], i2[["date", "pnl", "split", "year"]]])
    print("combined with I2 v1:"); print(split_table(comb).to_string())
    print("I2 alone:"); print(split_table(i2).to_string())
    print("daily pnl corr with I2 (days both trade):",
          round(tr.groupby("date").pnl.sum().to_frame("a").join(i2.groupby("date").pnl.sum().to_frame("b"), how="inner").corr().iloc[0, 1], 3))
    tr.to_csv(os.path.join(HERE, "results", f"trades_{name}.csv"), index=False)

# neighbours
print("=" * 30, "neighbours NQES_RS (time x lookback x hold)")
nb = []
for t in ["09:40", "09:45", "09:50"]:
    for lb in [30, 60, 90]:
        for h in [90, 120, 150]:
            tr = sim.run_signals(g, C.nqes_rs_signals(g, lv, t=t, lookback=lb, hold=h, es=es))
            nb.append({"time": t, "lookback": lb, "hold": h, **avg_by_split(tr)})
NB = pd.DataFrame(nb); print(NB.to_string())
print("disc>0:", (NB.discovery > 0).sum(), "/", len(NB), " val>0:", (NB.validation > 0).sum(), "/", len(NB))
NB.to_csv(os.path.join(HERE, "results", "neighbours_NQES_RS.csv"), index=False)

print("=" * 30, "neighbours MOM_OPEN (time x X x exit)")
nb = []
for t in ["09:55", "10:00", "10:05"]:
    for x in [60, 80, 100]:
        for ex in ["12:29", "15:59"]:
            tr = sim.run_signals(g, C.mom_open_signals(g, lv, t=t, x=x, exit_hhmm=ex))
            nb.append({"time": t, "x": x, "exit": ex, **avg_by_split(tr)})
NB = pd.DataFrame(nb); print(NB.to_string())
print("disc>0:", (NB.discovery > 0).sum(), "/", len(NB), " val>0:", (NB.validation > 0).sum(), "/", len(NB))
NB.to_csv(os.path.join(HERE, "results", "neighbours_MOM_OPEN.csv"), index=False)
