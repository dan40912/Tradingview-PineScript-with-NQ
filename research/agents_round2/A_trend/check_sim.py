"""核對：我的 sim1 與 lib/sim.py 在 I2 v1（+55/−145、12:29 出場）與移動停損上是否一致。"""
import time, numpy as np
from tlib import *
from lib import sim, strategies
D = Data()
S = strategies.i2_v1_signals(D.g, D.lv)
T0 = sim.run_signals(D.g, S)
sig = [dict(i=s["i"], e=s["entry_slot"] - 930, d=s["dir"]) for s in S]
T1 = run(D, sig, dict(stop_pts=145, tp_pts=55, xidx=179))
m = T0.merge(T1, left_on=["date"], right_on=["date"])
print("I2 筆數", len(T0), len(T1), "損益差最大", np.abs(m.pnl_x - m.pnl_y).max(), "合計", T0.pnl.sum(), T1.pnl.sum())
# 移動停損核對
S2 = [dict(i=x["i"], entry_slot=x["entry_slot"], dir=x["dir"]) for x in S]
T0 = sim.run_signals(D.g, S2, tp_pts=np.inf, sl_pts=100, trail_pts=60, trail_trigger=40, exit_slot=930 + 385)
T1 = run(D, sig, dict(stop_pts=100, trail=60, ttrig=40, xidx=385))
m = T0.merge(T1, on="date")
dif = np.abs(m.pnl_x - m.pnl_y); print("移動停損 差異筆數", (dif > 1e-6).sum(), "最大", dif.max(), T0.pnl.sum(), T1.pnl.sum())
t = time.time(); [run(D, sig, dict(stop_pts=100, trail=60, ttrig=40, xidx=385)) for _ in range(20)]; print("每次", (time.time() - t) / 20)
