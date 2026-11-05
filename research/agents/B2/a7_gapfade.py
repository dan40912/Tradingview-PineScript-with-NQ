"""A7: gap fade on NO-NEWS days (not CPI/NFP/FOMC) — neighbourhood + permutation, DISCOVERY only."""
from util import *
g, lv, rf, ev = load()
nd = ndays(g, "discovery")
ld = lv[lv.split == "discovery"]
t1 = ev.any_tier1.values; r830 = ev.rel830.values
atr = rf.atr20.values
rows = []; nvar = 0
for slm in [0.5, 0.75, 1.0, 1.5, 2.0]:
    for eidx in [1, 5, 15]:
        for ex in [1109, 1319]:
            for mn in [0.0, 0.1, 0.2]:
                nvar += 1
                s = [x for x in baselines.gap_fade(g, ld, sl_mult=slm, entry_idx=eidx, exit_slot=ex)
                     if abs(lv.open_0930[x["i"]] - lv.pdc[x["i"]]) >= mn * (atr[x["i"]] if np.isfinite(atr[x["i"]]) else 0)]
                tr = sim.run_signals(g, s)
                a = tr[~t1[tr.i]]; b = tr[t1[tr.i]]
                sa = metrics.summary(a, n_days=nd); h = half(a)
                rows.append({"sl": slm, "eidx": eidx, "exit": sim.hhmm_of(ex), "mingap_atr": mn, "n": sa["n"], "avg": sa["avg"],
                             "win": sa["win%"], "maxdd": sa["maxdd"], "ret/dd": sa["ret/dd"], "sh": sa["sharpe_ann"],
                             "D1": round(a.pnl[h == "D1"].mean()), "D2": round(a.pnl[h == "D2"].mean()), "D3": round(a.pnl[h == "D3"].mean()),
                             "tier1_avg": round(b.pnl.mean(), 0), "all_avg": round(tr.pnl.mean(), 0)})
R = pd.DataFrame(rows)
print(R.to_string())
print("variants", nvar, "share avg>0:", (R.avg > 0).mean(), "share>=50:", (R.avg >= 50).mean(),
      "all3>0:", ((R.D1 > 0) & (R.D2 > 0) & (R.D3 > 0)).mean(), "nonnews>tier1:", (R.avg > R.tier1_avg).mean())
