from util import *
g, lv, rf, ev = load()
nd = ndays(g, "discovery")
ld = lv[lv.split == "discovery"]
tr = attach(sim.run_signals(g, baselines.orb(g, ld, 30, tp_mult=1.0, sl_mult=1.0)), rf, ev)
a = tr[tr.rel830]
print(a[["date", "dir", "entry_time", "exit_time", "reason", "pts", "pnl", "or_w", "cpi", "nfp", "absgap_atr", "atr20"]].to_string())
print("excl Apr-2025:", a[~a.date.str.startswith("2025-04")].pnl.mean(), "excl top3:", a.pnl.sort_values().iloc[:-3].mean())
print("median pnl", a.pnl.median(), " long/short:", a.groupby("dir").pnl.agg(["count", "mean"]).to_dict())
print("cpi vs nfp:", a.groupby("cpi").pnl.agg(["count", "mean"]))
km = tr.rel830.values
for st in ["avg", "sharpe", "retdd"]:
    print(st, perm_test(tr, km, g, n=5000, stat=st))
# gap-control: within absgap terciles
cuts = np.quantile(tr.absgap_atr.dropna(), [1/3, 2/3])
tr["gb"] = np.searchsorted(cuts, tr.absgap_atr)
print(tr.groupby(["gb", "rel830"]).pnl.agg(["count", "mean"]).round(0))
cuts = np.quantile(tr.atr20_pct.dropna(), [1/3, 2/3]); tr["vb"] = np.searchsorted(cuts, tr.atr20_pct)
print(tr.groupby(["vb", "rel830"]).pnl.agg(["count", "mean"]).round(0))
# points (vol-free) per OR width
tr["R"] = tr.pts / tr.or_w
print("R-multiple mean rel/other:", tr.groupby("rel830").R.mean().round(3).to_dict())
