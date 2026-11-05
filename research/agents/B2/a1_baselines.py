"""A1: baseline strategies, overall stats (discovery focus)."""
from util import *
g, lv, rf, ev = load()
atr = rf.atr20.values
S = {
 "I2v1": strategies.i2_v1_signals(g, lv),
 "ORB30_1R": baselines.orb(g, lv, 30, tp_mult=1.0, sl_mult=1.0),
 "ORB30_2R_half": baselines.orb(g, lv, 30, tp_mult=1.0, sl_mult=0.5),
 "ORB15_1R": baselines.orb(g, lv, 15, tp_mult=1.0, sl_mult=1.0),
 "ORB15_eod": baselines.orb(g, lv, 15, tp_mult=2.0, sl_mult=1.0, exit_slot=1319),
 "GAPF_<0.5atr": baselines.gap_fade(g, lv, max_gap_atr=0.5, atr=atr),
 "GAPF_all": baselines.gap_fade(g, lv),
 "LONG0935": baselines.fixed_time(g, lv, 5, 1),
 "SHORT0935": baselines.fixed_time(g, lv, 5, -1),
 "LONG0935_noTP": baselines.fixed_time(g, lv, 5, 1, tp=np.inf, sl=np.inf),
}
out = {}
for k, s in S.items():
    tr = sim.run_signals(g, s)
    tr.to_pickle(f"/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/d4597cec-8294-562a-9e11-5568b74be6dd/scratchpad/tr_{k}.pkl")
    t = table(tr, g)
    print("==", k); print(t[t.index != "validation"].to_string())
