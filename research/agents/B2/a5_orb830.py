"""A5: ORB on 8:30-release days — parameter neighbourhood, gap-control, permutation (DISCOVERY only)."""
from util import *
SP = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/d4597cec-8294-562a-9e11-5568b74be6dd/scratchpad/"
g, lv, rf, ev = load()
nd = ndays(g, "discovery")
rel = ev.rel830.values
lv_d = lv[(lv.split == "discovery")]
rows = []; nvar = 0
for orm in [15, 20, 30, 45, 60]:
    for tpm in [0.75, 1.0, 1.5, 2.0]:
        for slm in [0.5, 0.75, 1.0]:
            for ex in [1109, 1319]:
                nvar += 1
                s = baselines.orb(g, lv_d, orm, tp_mult=tpm, sl_mult=slm, exit_slot=ex)
                tr = sim.run_signals(g, s)
                a = tr[rel[tr.i]]; b = tr[~rel[tr.i]]
                sa = metrics.summary(a, n_days=nd)
                h = half(a)
                rows.append({"or": orm, "tp": tpm, "sl": slm, "exit": sim.hhmm_of(ex), "n": sa["n"], "avg": sa["avg"],
                             "win": sa["win%"], "maxdd": sa["maxdd"], "ret/dd": sa["ret/dd"], "sh": sa["sharpe_ann"],
                             "D1": round(a.pnl[h == "D1"].mean()), "D2": round(a.pnl[h == "D2"].mean()), "D3": round(a.pnl[h == "D3"].mean()),
                             "other_avg": round(b.pnl.mean(), 0)})
R = pd.DataFrame(rows)
print(R.to_string())
print("variants", nvar, " share avg>0:", (R.avg > 0).mean(), " share avg>=50:", (R.avg >= 50).mean(),
      " share all-3-halves>0:", ((R.D1 > 0) & (R.D2 > 0) & (R.D3 > 0)).mean(), " share rel>other:", (R.avg > R.other_avg).mean())
R.to_pickle(SP + "a5.pkl")
