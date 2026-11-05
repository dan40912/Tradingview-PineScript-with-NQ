"""A8: (a) ORB30 by weekday on non-release days; (b) vol-normalised I2 exits; (c) gap-fade risk caps; DISCOVERY."""
from util import *
g, lv, rf, ev = load()
nd = ndays(g, "discovery")
ld = lv[lv.split == "discovery"]
atr = rf.atr20.values
# (a)
tr = attach(sim.run_signals(g, baselines.orb(g, ld, 30)), rf, ev)
print(tr[~tr.rel830].groupby("weekday").pnl.agg(["count", "mean"]).round(0))
print("FOMC days ORB:", tr[tr.fomc].pnl.agg(["count", "mean"]).round(0).to_dict())
# (b) vol-normalised I2: TP = a*ATR20, SL = b*ATR20 (median disc ATR20 ~ 326 -> 55/145 = 0.17/0.44)
base = strategies.i2_v1_signals(g, ld)
rows = []
for a in [0.12, 0.17, 0.22]:
    for b in [0.35, 0.45, 0.55]:
        for sized in [False, True]:
            s = []
            for x in base:
                A = atr[x["i"]]
                if not np.isfinite(A):
                    A = 326.0
                y = dict(x, tp_pts=a * A, sl_pts=b * A)
                if sized:
                    y["contracts"] = float(np.clip(np.round(326.0 / A, 1), 0.3, 2.0))
                s.append(y)
            t = sim.run_signals(g, s); sm = metrics.summary(t, n_days=nd); h = half(t)
            rows.append({"tp_k": a, "sl_k": b, "sized": sized, **{k: sm[k] for k in ["n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann"]},
                         "D1": round(t.pnl[h == "D1"].mean()), "D2": round(t.pnl[h == "D2"].mean()), "D3": round(t.pnl[h == "D3"].mean())})
print(pd.DataFrame(rows).to_string())
t = sim.run_signals(g, base); h = half(t)
print("I2 base halves:", [round(t.pnl[h == k].mean()) for k in ["D1", "D2", "D3"]])
# (c) gap fade non-news with caps
t1 = ev.any_tier1.values
rows = []
for cap in [None, 1.5, 1.0, 0.75]:
    for slpt_cap in [None, 150, 100]:
        s = []
        for x in baselines.gap_fade(g, ld, sl_mult=0.75, entry_idx=1, exit_slot=1109):
            i = x["i"]; gp = abs(lv.open_0930[i] - lv.pdc[i]); A = atr[i] if np.isfinite(atr[i]) else 326.0
            if t1[i] or gp < 0.1 * A or (cap is not None and gp > cap * A):
                continue
            if slpt_cap is not None:
                x = dict(x, sl_pts=min(x["sl_pts"], slpt_cap))
            s.append(x)
        tt = sim.run_signals(g, s); sm = metrics.summary(tt, n_days=nd); h = half(tt)
        rows.append({"gapcap_atr": cap, "slcap_pts": slpt_cap, **{k: sm[k] for k in ["n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann"]},
                     "worst": tt.pnl.min(), "D1": round(tt.pnl[h == "D1"].mean()), "D2": round(tt.pnl[h == "D2"].mean()), "D3": round(tt.pnl[h == "D3"].mean())})
print(pd.DataFrame(rows).to_string())
