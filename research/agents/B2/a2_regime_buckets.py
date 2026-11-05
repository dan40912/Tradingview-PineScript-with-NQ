"""A2: tercile buckets of pre-open regime features vs strategy P&L, DISCOVERY ONLY (validation hidden).
Consistency = sign(top - bottom tercile) identical in D1 (24H1), D2 (24H2), D3 (25H1)."""
from util import *
SP = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/d4597cec-8294-562a-9e11-5568b74be6dd/scratchpad/"
g, lv, rf, ev = load()
FEATS = ["atr20", "atr20_px", "atr20_pct", "atr5_20", "rv_ratio", "on_range_atr", "absgap_atr", "va_width_atr",
         "prev_range_atr", "volvol20", "ret5", "ret20", "trend_eff20", "dist_ma20_pct", "dd60_pct", "fear", "fear_pct"]
STRATS = ["I2v1", "ORB30_1R", "ORB15_eod", "GAPF_all", "LONG0935_noTP"]
nvar = 0
summary_rows = []
for s in STRATS:
    tr = attach(pd.read_pickle(SP + f"tr_{s}.pkl"), rf, ev)
    tr = tr[tr.split == "discovery"]
    for f in FEATS:
        nvar += 1
        bt, cuts = bucket_table(tr, f)
        d = {h: bt[f"{h}_avg"].values for h in ["D1", "D2", "D3"]}
        signs = [np.sign(d[h][-1] - d[h][0]) for h in d]
        mono = all(((np.diff(d[h]) > 0).all() or (np.diff(d[h]) < 0).all()) for h in d) and len(set(signs)) == 1
        summary_rows.append({"strat": s, "feat": f, "lo": bt.disc_avg.values[0], "mid": bt.disc_avg.values[1],
                             "hi": bt.disc_avg.values[-1], "consistent": len(set(signs)) == 1, "monotone3": mono,
                             "D1": d["D1"].round(0).tolist(), "D2": d["D2"].round(0).tolist(), "D3": d["D3"].round(0).tolist(),
                             "cuts": np.round(cuts, 2).tolist()})
R = pd.DataFrame(summary_rows)
print(R.to_string())
print("tables:", nvar, " consistent:", R.consistent.sum(), " expected by chance (1/4 each):", nvar / 4, " monotone:", R.monotone3.sum())
R.to_pickle(SP + "a2.pkl")
