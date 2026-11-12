"""勝率研究：所有候選進場點＋進場當下的條件，5 種止盈（停損 145）各算一次。研究用途，不是投資建議。"""
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib 和 research/multi_entry 的 family.py、f_family.py；放到該分支 research/ 下執行（見 docs/22）。
import sys, os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import f_family as FF
from lib import sim, strategies
from lib.data import RTH_OFF as R0
g, lv = FF.NQ.g, FF.NQ.lv; ge, lve = FF.ES.g, FF.ES.lv
L = lv.set_index("i"); LE = lve.set_index("date")
# 波動：前 10 個交易日美股盤中振幅平均 / 2024-25 中位數
rng = pd.Series(np.nanmax(g.rth(g.H), 1) - np.nanmin(g.rth(g.L), 1), index=g.dates)
atr = rng[g.tradable].rolling(10).mean().shift(1).reindex(g.dates).ffill()
REF = float(atr[(atr.index >= "2024-01-17") & (atr.index <= "2025-12-31")].median())
# 候選訊號
S = [dict(s, exit_slot=R0 + 179, rule="I2", side="空" if s["dir"] == -1 else "多") for s in strategies.i2_v1_signals(g, lv)]
SN = FF.SN.copy(); SE = FF.SE[["date", "rule", "side", "entry_slot"]].rename(columns={"entry_slot": "es_slot"})
SN = SN.merge(SE, on=["date", "rule", "side"], how="left")
for r in SN.itertuples():
    S.append(dict(i=r.i, entry_slot=r.entry_slot, dir=r.dir, exit_slot=r.exit_slot, rule=r.rule, side=r.side,
                  es_ok=bool(pd.notna(r.es_slot) and r.es_slot <= r.entry_slot)))
rows = []
for s in S:
    i, e, d = s["i"], s["entry_slot"], s["dir"]; date = g.dates[i]; r = L.loc[i]
    px = g.O[i][e] if not np.isnan(g.O[i][e]) else np.nan
    f = dict(date=date, rule=s["rule"], side=s["side"], dir=d, entry_slot=e, exit_slot=s["exit_slot"])
    f["es_ok"] = s.get("es_ok", None)
    if s["rule"] == "I2":
        f["es_ok"] = bool(date in LE.index and LE.loc[date].i2_dir == r.i2_dir)
    f["i2"] = "無方向" if r.i2_dir == 0 else ("順" if r.i2_dir == d else "逆")
    f["vol"] = atr.get(date, np.nan) / REF
    f["wd"] = pd.Timestamp(date).weekday()
    f["hour"] = int(((e + 18 * 60) % 1440) // 60)
    # 進場價相對前一日收盤：和交易方向同向（價格已經往這邊走）或反向
    f["move"] = d * (px - r.pdc) if pd.notna(r.pdc) and pd.notna(px) else np.nan
    # 進場價在前一日價值區的哪裡
    f["va_pos"] = "區內" if r.val <= px <= r.vah else ("上方" if px > r.vah else "下方") if pd.notna(px) else None
    # 美盤：進場時價格相對當天 VWAP（往 VWAP 方向做＝回歸）
    if e >= R0:
        H, Lw, C, V = g.H[i][R0:e], g.L[i][R0:e], g.C[i][R0:e], g.V[i][R0:e]
        m = np.isfinite(C) & np.isfinite(V)
        vw = np.sum((H[m] + Lw[m] + C[m]) / 3 * V[m]) / max(np.sum(V[m]), 1) if m.any() else np.nan
        f["to_vwap"] = bool(d * (vw - px) > 0) if pd.notna(vw) else None
    else:
        f["to_vwap"] = None
    f["on_range"] = (r.onh - r.onl) / atr.get(date, np.nan) if pd.notna(r.onh) else np.nan
    rows.append(f)
F = pd.DataFrame(rows)
for tp in (20, 30, 40, 55, 80):
    T = sim.run_signals(g, [dict(i=s["i"], entry_slot=s["entry_slot"], dir=s["dir"], exit_slot=s["exit_slot"], k=k) for k, s in enumerate(S)], tp_pts=float(tp), sl_pts=145.0)
    T = T.set_index("k")
    F[f"pnl{tp}"] = T.pnl.reindex(range(len(S))).values
    F[f"why{tp}"] = T.reason.reindex(range(len(S))).values
F.to_pickle("wr_table.pkl"); print(len(F), F.date.max(), F.groupby("rule").size().head(50).to_dict())
