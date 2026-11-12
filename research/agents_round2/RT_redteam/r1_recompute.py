"""檢查 2：獨立重算 C1、TD10(q80 k3)，和 A、C 的交易表逐筆比對。"""
import os, numpy as np, pandas as pd
import rt_core as R
A_DIR = os.path.join(os.path.dirname(R.OUT), "A_trend"); C_DIR = os.path.join(os.path.dirname(R.OUT), "C_indicators")
print("壞價 K 棒：", R.BAD)
print("ADR10 2026-11-11 =", round(R.ADR10[R.g.dindex["2026-11-11"]], 1))
# ---- C1 ----
c1 = R.run(R.c1_signals())
c1.to_csv(os.path.join(R.OUT, "rt_trades_C1.csv"), index=False)
print(R.table(c1, "RT C1").to_string(index=False))
a = pd.read_csv(os.path.join(A_DIR, "trades_C1_開盤驅動5分.csv"))
print("A 的 C1：", len(a), "筆 合計", a.pnl.sum().round(), " RT：", len(c1), "筆 合計", c1.pnl.sum().round())
m = c1.merge(a[["date", "d", "entry", "exit", "pnl"]], on="date", how="outer", suffixes=("", "_A"), indicator=True)
print("配對：", m._merge.value_counts().to_dict())
both = m[m._merge == "both"]
diff = both[(both.pnl - both.pnl_A).abs() > 1]
print("同一天但損益不同：", len(diff)); print(diff[["date", "d", "d_A", "entry", "entry_A", "exit", "exit_A", "pnl", "pnl_A", "why"]].to_string())
print(m[m._merge != "both"][["date", "_merge", "pnl", "pnl_A"]].to_string())
# ---- TD10 ----
F = R.td_features(960, adjust=True)
F.to_csv(os.path.join(R.OUT, "rt_td_features.csv"), index=False)
dv, thr = R.td_signal(F, 0.8, 3)
print("TD10 q80 門檻（自己用探索期算）：", {k: round(v, 3) for k, v in thr.items()}, " C 報告：0.322/0.800/1.300/21.74/0.382")
td = R.run(R.td_sigs(dv))
td.to_csv(os.path.join(R.OUT, "rt_trades_TD10.csv"), index=False)
print(R.table(td, "RT TD10 q80k3").to_string(index=False))
t = td.pnl.values; print("全期 t =", round(t.mean() / (t.std(ddof=1) / np.sqrt(len(t))), 2))
# C 的 TD 單：從 C 的候選 1 交易表取 pt==TD順勢
c = pd.read_csv(os.path.join(C_DIR, "trades_候選1.csv"))
ctd = c[c.pt == "TD順勢"]
print("C 的候選1 裡的 TD 單（已過風控）：", len(ctd), "筆")
m = td.merge(ctd[["date", "dir", "pnl"]], on="date", how="outer", suffixes=("", "_C"), indicator=True)
print("配對：", m._merge.value_counts().to_dict())
bb = m[m._merge == "both"]; print("同一天損益不同：", int(((bb.pnl - bb.pnl_C).abs() > 1).sum()), " 方向不同：", int((bb.d != bb.dir).sum()))
print("只有 RT 有（C 被風控擋或判斷不同）：", m[m._merge == "left_only"][["date", "d", "pnl"]].to_string())
print("只有 C 有：", m[m._merge == "right_only"][["date", "dir", "pnl_C"]].to_string())
# 無換月調整的版本
F2 = R.td_features(960, adjust=False); dv2, _ = R.td_signal(F2, 0.8, 3)
print("不做換月調整時訊號不同的天數：", int((dv2 != dv).sum()))
td2 = R.run(R.td_sigs(dv2)); print(R.line(td2, "TD10 不做換月調整"))
