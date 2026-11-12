"""(d) 用 B 實際網格裡的攤平／加碼計畫，檢查實際單筆虧損是否超過 B 算的最壞虧損（import B 的 core）。"""
import os, sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/agents/B_exits")
import core as BC
pl = []
for x, s, sl, tp in itertools.product([30, 50, 75, 100], [0.5, 1.0], [145, 200], [55, 30]):
    if x < sl: pl.append(("D攤平1", BC.plan(sl=sl, tp1=tp, a1d=-x, a1s=s, tp_avg=1)))
for x, sl, tp in itertools.product([40, 60, 80, 100, 120], [145, 175, 200, 240], [30, 55]):
    if x < sl: pl.append(("D攤平半半", BC.plan(sl=sl, tp1=tp, a1d=-x, a1s=1.0, tp_avg=1)))
for x1, x2, sl, tp in itertools.product([40, 60], [90, 120], [160, 200], [30, 55]):
    pl.append(("D三段", BC.plan(sl=sl, tp1=tp, a1d=-x1, a1s=0.75, a2d=-x2, a2s=0.75, tp_avg=1)))
for x, sl in itertools.product([50, 75, 100], [145, 200]):
    pl.append(("D救援", BC.plan(sl=sl, tp1=55, a1d=-x, a1s=1.0, tp_after_add=0.0)))
for x, r, po, tp in itertools.product([20, 30, 40], [0.5, 1.0], [-30, 0, 5], [55, 80, 110]):
    if tp > x: pl.append(("E加碼", BC.plan(tp1=tp, a1d=x, a1s=r, pyr_off=po)))
for x, r, sl, tp in itertools.product([30, 50, 75, 100, 125], [0.5, 1.0, 2.0], [125, 145, 175, 200, 250], [30, 55]):
    if x < sl - 20: pl.append(("D同風險", BC.plan(sl=sl, tp1=tp, a1d=-x, a1s=r, tp_avg=1)))
res = []
for nm, p in pl:
    R = BC.simulate(p); ex = -R.pnl1 - R.worst1
    res.append(dict(fam=nm, viol=int((ex > 1e-6).sum()), max_over=ex.max(), n=len(R)))
Z = pd.DataFrame(res)
print(Z.groupby("fam").agg(plans=("n", "size"), plans_viol=("viol", lambda v: int((v > 0).sum())), trades_viol=("viol", "sum"), max_over=("max_over", "max")).to_string())
