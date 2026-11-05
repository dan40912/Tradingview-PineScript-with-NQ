"""Final evaluation of FROZEN candidates on discovery and validation (no holdout).
Writes eval_results.json and prints tables."""
import warnings, json; warnings.filterwarnings("ignore")
from a1lib import *
from candidates import CANDIDATES, oov_orb
out = {}
pd.set_option("display.width", 250)
for sym in ["NQ", "ES"]:
    g, lv = load(sym)
    i2 = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
    i2d = set(i2.date)
    res = {"I2v1": i2}
    for name, fn in CANDIDATES.items():
        res[name] = sim.run_signals(g, fn(g, lv))
    print(f"\n######## {sym}")
    for name, tr in res.items():
        print(f"\n== {name}")
        t = table(tr, g); print(t[["n","win%","avg","total","pf","maxdd","ret/dd","sharpe_ann","t"]].to_string())
        out[f"{sym}|{name}|table"] = t.astype(str).to_dict()
        if name == "I2v1":
            continue
        for sp in ["discovery", "validation"]:
            x = tr[tr.split == sp]
            on, off = x[x.date.isin(i2d)], x[~x.date.isin(i2d)]
            m = on.merge(i2[["date", "dir"]], on="date", suffixes=("", "_i2"))
            same = m[m.dir == m.dir_i2]; opp = m[m.dir != m.dir_i2]
            print(f"  {sp}: on-I2 days n={len(on)} avg={on.pnl.mean():.0f} tot={on.pnl.sum():.0f} | off-I2 n={len(off)} avg={off.pnl.mean():.0f} tot={off.pnl.sum():.0f}"
                  f" | same-dir {len(same)} avg {same.pnl.mean():.0f}, opp-dir {len(opp)} avg {opp.pnl.mean():.0f}"
                  f" | long {x[x.dir==1].pnl.mean():.0f} ({(x.dir==1).sum()}) short {x[x.dir==-1].pnl.mean():.0f} ({(x.dir==-1).sum()})")
            days = g.dates[g.tradable & (g.split == sp)]
            a = x.groupby("date").pnl.sum().reindex(days, fill_value=0)
            b = i2[i2.split == sp].groupby("date").pnl.sum().reindex(days, fill_value=0)
            c = a + b
            print(f"     daily corr w/ I2 {a.corr(b):+.2f}; I2+cand: total {c.sum():.0f} maxdd {metrics.max_drawdown(c.values):.0f} sharpe {c.mean()/c.std()*np.sqrt(252):.2f}")
            out[f"{sym}|{name}|{sp}|overlap"] = dict(on_n=len(on), on_avg=float(on.pnl.mean()) if len(on) else None, off_n=len(off),
                                                    off_avg=float(off.pnl.mean()) if len(off) else None, corr=float(a.corr(b)),
                                                    comb_total=float(c.sum()), comb_dd=float(metrics.max_drawdown(c.values)),
                                                    comb_sharpe=float(c.mean()/c.std()*np.sqrt(252)))
        v = tr[tr.split == "validation"]
        if len(v):
            v = v.copy(); v["m"] = v.date.str[:7]
            print("  validation by month:", v.groupby("m").pnl.sum().round(0).to_dict())
        print("  exit reasons (val):", v.reason.value_counts().to_dict() if len(v) else {})
# informational: neighbour grid on validation (NOT used to change anything)
g, lv = load("NQ")
base = {"OOV_ORB15_EOD": dict(N=15, cutoff="12:00", stop_mult=1.0, k=None, exit_time="15:59"),
        "OOV_ORB30_1R": dict(N=30, cutoff="12:00", stop_mult=1.0, k=1.0, exit_time="15:59")}
grid = {"N": {"OOV_ORB15_EOD": [10, 20], "OOV_ORB30_1R": [20, 45]}, "cutoff": ["11:00", "13:00"], "stop_mult": [0.75, 1.25],
        "k": {"OOV_ORB15_EOD": [3.0], "OOV_ORB30_1R": [0.75, 1.5]}, "exit_time": ["15:00", "15:30"]}
print("\n== validation neighbours (informational)")
for cn, b in base.items():
    for p, vals in grid.items():
        vals = vals[cn] if isinstance(vals, dict) else vals
        for vv in vals:
            kw = dict(b); kw[p] = vv
            tr = sim.run_signals(g, oov_orb(g, lv, **kw))
            sd, sv = summ(tr, g, "discovery"), summ(tr, g, "validation")
            print(f"  {cn} {p}={vv}: disc avg {sd['avg']} sh {sd['sharpe_ann']} | val n {sv['n']} avg {sv['avg']} dd {sv['maxdd']} sh {sv['sharpe_ann']}")
json.dump(out, open("eval_results.json", "w"), indent=1, default=str)
