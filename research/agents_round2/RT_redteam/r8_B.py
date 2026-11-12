"""B_exits 審查：(a) 記帳差異（已在 r5_combo 用自己的程式重現）；(b) 止損×口數高原（自己的模擬器）；
(d) DCA／加碼最壞虧損：用 B 的 core 跑它自己的計畫，檢查有沒有任何一筆實際虧損超過它算的最壞虧損（這部分刻意 import B 的程式）。"""
import os, sys, numpy as np, pandas as pd
import rt_core as R
out = []
def P(s=""): print(s); out.append(str(s))
# ---- (b) 高原 ----
rows = []
for sl in [145, 160, 175, 190, 200, 210, 225, 250, 275, 300]:
    Bt = R.base_trades(tp=55.0, sl=float(sl))
    for size in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if (sl * 20 + 48) * size > 3000:
            continue
        X = Bt.copy(); X["pnl"] *= size; X["worst"] = (sl * 20 + 48) * size
        Bk = R.book(X, "real")
        r = dict(sl=sl, size=size)
        for p in R.PER + ["全期"]:
            x = Bk if p == "全期" else Bk[Bk.P == p]
            s = R.stats(x); r[p] = s["tot"]; r[p + "_dd"] = s["mdd"]; r[p + "_比"] = round(s["tot"] / -s["mdd"], 2) if s["mdd"] < 0 else np.inf
        rows.append(r)
Z = pd.DataFrame(rows); Z.to_csv(os.path.join(R.OUT, "rt_B_plateau.csv"), index=False)
P("== B 方案二高原（+55 止盈，Pine 式記帳，每日 $3,000）：各段合計／報酬回撤比")
P(Z[["sl", "size", "探索", "探索_比", "驗證", "驗證_比", "測試", "測試_比", "近期", "全期", "全期_dd", "全期_比"]].to_string(index=False))
base = Z[(Z.sl == 145) & (Z["size"] == 1.0)].iloc[0]
win = Z[(Z.探索_比 > base.探索_比) & (Z.驗證_比 > base.驗證_比) & (Z.測試_比 > base.測試_比)]
P(f"三段報酬回撤比都比 145×1 口好的格：{len(win)}／{len(Z)}：{win[['sl','size']].values.tolist()}")
# 同樣「每口最壞虧損」約 $2,000～2,950 的比較：等風險
P("\n== 每筆最壞虧損相近時（$2,000～$3,000），各段報酬回撤比")
Z["worst"] = (Z.sl * 20 + 48) * Z["size"]
P(Z[(Z.worst >= 2000)].sort_values("worst")[["sl", "size", "worst", "探索_比", "驗證_比", "測試_比", "全期", "全期_比"]].to_string(index=False))
# ---- (d) DCA 最壞虧損 ----
sys.path.insert(0, os.path.join(os.path.dirname(R.OUT), "B_exits"))
import core as BC
plans = []
for sl in [145, 200, 250]:
    for x in [20, 30, 50, 80]:
        for s in [0.5, 1.0]:
            plans.append(("攤平", BC.plan(sl=sl, tp1=55, a1d=-x, a1s=s, tp_avg=1)))
            plans.append(("兩段攤平", BC.plan(sl=sl, tp1=55, a1d=-x, a1s=s, a2d=-2 * x, a2s=s, tp_avg=1)))
            plans.append(("救援", BC.plan(sl=sl, tp1=55, a1d=-x, a1s=s, tp_after_add=0.0)))
for x in [20, 40, 60]:
    for po in [-145, -50, 0, 10]:
        plans.append(("順勢加碼", BC.plan(tp1=110, a1d=x, a1s=1.0, pyr_off=po)))
bad = 0; worst_gap = []
for nm, p in plans:
    Rr = BC.simulate(p)
    ex = (-Rr.pnl1 - Rr.worst1)
    worst_gap.append(ex.max())
    if (ex > 1e-6).any():
        bad += 1
        k = ex.idxmax(); P(f"  超過最壞虧損：{nm} {p[[0,1,11,12,13,14,16]].tolist()} 最多超過 ${ex.max():.0f}（{Rr.loc[k,'date']}，損益 {Rr.loc[k,'pnl1']:.0f}，算的最壞 {Rr.loc[k,'worst1']:.0f}）")
P(f"(d) 測了 {len(plans)} 個 B 的攤平／加碼計畫：實際單筆虧損超過 B 計算之最壞虧損的計畫 {bad} 個；最接近的差距 ${max(worst_gap):+.0f}（負值＝沒超過）")
# 方案二本身：每筆實際虧損 vs 最壞
R2 = BC.simulate(BC.plan(sl=200, tp1=55)); P(f"方案二 SL200：最大單筆虧損（1 口）{R2.pnl1.min():.0f}，算的最壞 {R2.worst1.max():.0f}")
open(os.path.join(R.OUT, "rt_B.txt"), "w").write("\n".join(out))
