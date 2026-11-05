"""Check that the shared engine reproduces the frozen I2 v1 trade list (backtest/i2_v1_trades.csv)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
from lib import data, features, sim, strategies, metrics

g = data.load_grid("NQ", include_holdout=True)
lv = features.daily_levels(g)
tr = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
ref = pd.read_csv(os.path.join(data.ROOT, "backtest", "i2_v1_trades.csv"), encoding="utf-8-sig")
ref = ref[ref.date <= g.dates[-1]]
m = ref.merge(tr, on="date", how="outer", indicator=True, suffixes=("_ref", ""))
print("ref", len(ref), "ours", len(tr), m._merge.value_counts().to_dict())
both = m[m._merge == "both"]
bad = both[(both.entry_ref != both.entry) | (both.entry_time_et != both.entry_time) | ((both.net_usd - both.pnl).abs() > 0.01)]
print("mismatches:", len(bad))
print(bad[["date", "entry_time_et", "entry_time", "entry_ref", "entry", "net_usd", "pnl", "reason_ref", "reason"]].head(20))
lvchk = ref.merge(lv, on="date")
print("VA mismatches:", ((lvchk.val_x != lvchk.val_y) | (lvchk.vah_x != lvchk.vah_y) | (lvchk.poc_x != lvchk.poc_y)).sum())
print(metrics.by_period(tr))
print(m[m._merge != "both"][["date", "_merge", "entry_time_et", "entry_time", "contract_ref" if "contract_ref" in m else "contract"]])
for d in m[m._merge != "both"].date:
    r = lv[lv.date == d].iloc[0]
    print(d, r.quality, r.contract, r.has_prev, r.val, r.vah, r.open_0930, "prev:", lv.iloc[r.i-1].date, lv.iloc[r.i-1].quality)
print(lvchk[(lvchk.val_x != lvchk.val_y) | (lvchk.vah_x != lvchk.vah_y)][["date","val_x","val_y","vah_x","vah_y","poc_x","poc_y"]])
