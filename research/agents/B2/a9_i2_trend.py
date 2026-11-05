"""A9: I2 vs trend regime (economic prior: mean reversion suffers in efficient trends). DISCOVERY only."""
from util import *
SP = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/d4597cec-8294-562a-9e11-5568b74be6dd/scratchpad/"
g, lv, rf, ev = load()
nd = ndays(g, "discovery")
tr = attach(pd.read_pickle(SP + "tr_I2v1.pkl"), rf, ev)
tr = tr[tr.split == "discovery"].reset_index(drop=True)
tr["h"] = half(tr)
for col in ["ret20", "ret5", "dist_ma20_pct"]:
    tr["with_" + col] = np.sign(tr[col]) == tr.dir
    print(col, tr.groupby(["with_" + col, "h"]).pnl.agg(["count", "mean"]).round(0).unstack())
# quintiles of trend_eff20
for q in [(0.2, 0.4, 0.6, 0.8)]:
    bt, cuts = bucket_table(tr, "trend_eff20", q=q); print(bt[["D1_n", "D1_avg", "D2_n", "D2_avg", "D3_n", "D3_avg", "disc_n", "disc_avg", "disc_time%"]], cuts)
rows = []
for th in [0.25, 0.3, 0.35, 0.4, 0.45]:
    keep = ~(tr.trend_eff20 > th).values
    t = tr[keep]; sm = metrics.summary(t, n_days=nd)
    obs, p = perm_test(tr, keep, g, n=3000)
    rows.append({"skip_eff>": th, "n": sm["n"], "avg": sm["avg"], "maxdd": sm["maxdd"], "ret/dd": sm["ret/dd"], "sh": sm["sharpe_ann"],
                 "removed_avg": round(tr[~keep].pnl.mean()), "p_avg": round(p, 3),
                 "D123": [round(t.pnl[t.h == k].mean()) for k in ["D1", "D2", "D3"]]})
print(pd.DataFrame(rows).to_string())
# ORB on non-release days by trend_eff (supporting regime story)
o = attach(pd.read_pickle(SP + "tr_ORB30_1R.pkl"), rf, ev); o = o[(o.split == "discovery")]
bt, cuts = bucket_table(o[~o.rel830], "trend_eff20", q=(0.2, 0.4, 0.6, 0.8)); print("ORB non-830\n", bt[["D1_avg", "D2_avg", "D3_avg", "disc_n", "disc_avg"]])
