"""Volatility-regime proxy for the 2026 holdout (NQ price and range are larger in 2026; prior docs note the fixed
55/145 bracket became relatively small). Splits 2024-25 trades by the CAUSAL ATR14 (mean RTH range of the previous 14
full sessions) tercile. Fixed-point stops/targets that only work in low-ATR terciles are at risk in 2026."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_common as A
from lib import strategies, sim

g, lv = A.load("NQ")
C = A.all_candidates()
rng_ = np.array([np.nanmax(g.H[i, 930:1320]) - np.nanmin(g.L[i, 930:1320]) if g.has_rth[i] else np.nan for i in range(len(g))])
atr, hist = np.full(len(g), np.nan), []
for i in range(len(g)):
    if len(hist) >= 14:
        atr[i] = np.mean(hist[-14:])
    if g.has_rth[i] and not g.equity_holiday[i]:
        hist.append(rng_[i])
date_atr = dict(zip(g.dates, atr))
cuts = np.nanquantile(atr[g.tradable], [1 / 3, 2 / 3])
print("ATR14 tercile cuts (pts):", cuts.round(0), " | 2025H2 mean ATR14:",
      round(np.nanmean(atr[(g.split == 'validation') & g.tradable]), 0), " 2024 mean:",
      round(np.nanmean(atr[(g.year == '2024') & g.tradable]), 0))
rows = []
TR = {n: A.trades(n, C) for n in C}
TR["BASE/I2v1"] = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
for n, tr in TR.items():
    t = tr.assign(atr=tr.date.map(date_atr))
    t["terc"] = np.searchsorted(cuts, t.atr)
    rec = {"cand": n}
    for k, lab in enumerate(["lowATR", "midATR", "highATR"]):
        x = t[t.terc == k]
        rec[f"{lab}_n"] = len(x)
        rec[f"{lab}_avg"] = round(x.pnl.mean(), 0) if len(x) else np.nan
        if "reason" in x:
            rec[f"{lab}_stop%"] = round(100 * x.reason.isin(["sl", "stop"]).mean(), 0) if len(x) else np.nan
    rows.append(rec)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(A.HERE, "out_07_vol_regime.csv"), index=False)
with pd.option_context("display.width", 220):
    print(d.to_string(index=False))
