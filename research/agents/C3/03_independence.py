"""C3 step 3: is ES I2 an independent stream from NQ I2? (descriptive; computed per split)
Streams: NQ I2 v1 (55/145), NQ A3_C3 (POC/60/NP30), ES I2 bin5 shadow 22/58, ES I2 bin5 fixed 40/15."""
import sys, os, numpy as np, pandas as pd
from c3lib import data, features, levels, i2_exit_sigs, run, stats, tradable_days, daily, strategies
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "A3"))
from candidates import c3_poc_sl60_np30  # A3 frozen candidate (read-only import)

gN = data.load_grid("NQ"); lvN = features.daily_levels(gN)
gE = data.load_grid("ES"); lvE = levels(gE, 5.0)
S = {
    "NQ_I2v1": run(gN, strategies.i2_v1_signals(gN, lvN)),
    "NQ_A3C3": run(gN, c3_poc_sl60_np30(gN, lvN)),
    "ES_I2_22_58": run(gE, i2_exit_sigs(gE, lvE, tp=22, sl=58)),
    "ES_I2_40_15": run(gE, i2_exit_sigs(gE, lvE, tp=40, sl=15)),
}
for split in ["discovery", "validation"]:
    days = sorted(set(tradable_days(gN, split)) & set(tradable_days(gE, split)))
    dv = pd.DataFrame({k: daily(v, days) for k, v in S.items()})
    print(f"\n===== {split}: {len(days)} common tradable days")
    print("daily P&L correlation (all days, flat=0):"); print(dv.corr().round(2))
    both = (dv.NQ_I2v1 != 0) & (dv.ES_I2_22_58 != 0)
    print("days traded: NQ", int((dv.NQ_I2v1 != 0).sum()), " ES", int((dv.ES_I2_22_58 != 0).sum()), " both", int(both.sum()))
    print("corr on both-days:", dv[both].corr().round(2).loc["NQ_I2v1"].to_dict())
    n = S["NQ_I2v1"].set_index("date"); e = S["ES_I2_22_58"].set_index("date")
    com = n.index.intersection(e.index)
    com = [d for d in com if d in days]
    agree = (n.loc[com, "dir"] == e.loc[com, "dir"])
    print(f"same-day direction agreement: {agree.sum()}/{len(com)}")
    # entry-time gap
    gap = (n.loc[com, "entry_slot"] - e.loc[com, "entry_slot"]).abs()
    print("median |entry slot gap| (min):", float(gap.median()))
    # performance split by overlap
    for k in ["ES_I2_22_58", "ES_I2_40_15"]:
        t = S[k]; t = t[t.date.isin(days)]
        on = t.date.isin(set(S["NQ_I2v1"].date))
        print(f"{k}: on NQ-I2 days n={on.sum()} avg={t.pnl[on].mean():.0f} | off n={(~on).sum()} avg={t.pnl[~on].mean():.0f}")
    t = S["NQ_I2v1"]; t = t[t.date.isin(days)]; on = t.date.isin(set(S["ES_I2_22_58"].date))
    print(f"NQ_I2v1: on ES-I2 days n={on.sum()} avg={t.pnl[on].mean():.0f} | off n={(~on).sum()} avg={t.pnl[~on].mean():.0f}")
    # NQ state on ES-I2 days
    lvn = lvN.set_index("date"); lve = lvE.set_index("date")
    tab = pd.crosstab(lvn.loc[days, "i2_dir"].rename("NQ open vs own VA"), lve.loc[days, "i2_dir"].rename("ES open vs own VA"))
    print(tab)
