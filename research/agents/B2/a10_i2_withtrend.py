"""A10: I2 only in the direction of the prevailing trend — neighbourhood + permutation. DISCOVERY only."""
from util import *
SP = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/d4597cec-8294-562a-9e11-5568b74be6dd/scratchpad/"
g, lv, rf, ev = load()
nd = ndays(g, "discovery")
tr = attach(pd.read_pickle(SP + "tr_I2v1.pkl"), rf, ev)
tr = tr[tr.split == "discovery"].reset_index(drop=True); tr["h"] = half(tr)
rows = []
for w in [5, 10, 15, 20, 30, 40, 60]:
    for band in [0.0, 0.5, 1.0]:
        r = tr[f"ret{w}"].values
        withm = (np.sign(r) == tr.dir.values) & (np.abs(r) >= band)
        against = (np.sign(r) == -tr.dir.values) & (np.abs(r) >= band)
        for mode in ["skip_against", "half_against"]:
            wgt = np.where(against, 0.0 if mode == "skip_against" else 0.5, 1.0)
            t = tr.assign(pnl=tr.pnl * wgt)[wgt > 0]
            sm = metrics.summary(t, n_days=nd)
            p = perm_test(tr, wgt > 0, g, n=2000)[1] if mode == "skip_against" else np.nan
            rows.append({"lookback": w, "band%": band, "mode": mode, "n": sm["n"], "avg/unit": round(t.pnl.sum() / wgt[wgt > 0].sum()),
                         "total": sm["total"], "maxdd": sm["maxdd"], "ret/dd": sm["ret/dd"], "sh": sm["sharpe_ann"],
                         "with_avg": round(tr.pnl[withm].mean()), "against_avg": round(tr.pnl[against].mean()), "n_against": int(against.sum()),
                         "p_perm": round(p, 3) if p == p else "",
                         "D123_with": [round(tr.pnl[withm & (tr.h == k).values].mean()) for k in ["D1", "D2", "D3"]],
                         "D123_against": [round(tr.pnl[against & (tr.h == k).values].mean()) for k in ["D1", "D2", "D3"]]})
R = pd.DataFrame(rows)
print(R.to_string())
sk = R[R["mode"] == "skip_against"]
print("variants", len(R), " with>against share:", (sk.with_avg > sk.against_avg).mean(), " p<0.05 share:", (sk.p_perm.astype(float) < 0.05).mean())
# by direction
for w in [20]:
    tr["with"] = np.sign(tr[f"ret{w}"]) == tr.dir
    print(tr.groupby(["dir", "with"]).pnl.agg(["count", "mean"]).round(0))
    print(tr.groupby(["with"]).reason.value_counts().unstack())
