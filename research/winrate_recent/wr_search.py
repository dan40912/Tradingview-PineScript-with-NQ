"""勝率研究：近期（2026-10-01～11-11）找，過去（2024-01～2026-09）驗證。研究用途，不是投資建議。"""
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib 和 research/multi_entry 的 family.py、f_family.py；放到該分支 research/ 下執行（見 docs/22）。
import numpy as np, pandas as pd, itertools
F = pd.read_pickle("wr_table.pkl")
F["key"] = F.rule + "|" + F.side
F["W"] = np.where(F.date >= "2026-10-01", "REC", "HIST")
# 避免偷看未來：美股開盤（slot 930）前進場的交易，不能用 I2 方向、夜盤區間（要等 09:30 才知道）
pre = F.entry_slot < 930
F.loc[pre, "i2"] = None; F.loc[pre, "on_range"] = np.nan
# 進場當下就知道的：價格在前日價值區哪一邊 × 交易方向
F["vp"] = F.va_pos.astype(str) + np.where(F.dir == 1, "做多", "做空")
F["volb"] = pd.cut(F.vol, [0, 1.2, 1.8, 99], labels=["低波動<1.2", "中1.2-1.8", "高>1.8"]).astype(str)
F["moveb"] = np.where(F.move > 0, "已往這邊走", np.where(F.move < 0, "反方向走過", "—"))
F["onb"] = pd.cut(F.on_range, [0, 0.5, 0.8, 99], labels=["夜盤窄", "夜盤中", "夜盤寬"]).astype(str)
F["wdn"] = F.wd.map({0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 6: "日"})
FILT = {"ES同步": ("es_ok", [True]), "I2": ("i2", ["順", "無方向", "逆"]), "波動": ("volb", ["低波動<1.2", "中1.2-1.8", "高>1.8"]),
        "往VWAP": ("to_vwap", [True, False]), "走勢": ("moveb", ["已往這邊走", "反方向走過"]), "價值區": ("va_pos", ["區內", "上方", "下方"]),
        "夜盤區間": ("onb", ["夜盤窄", "夜盤中", "夜盤寬"]), "星期": ("wdn", ["一", "二", "三", "四", "五"]),
        "位置×方向": ("vp", ["上方做多", "上方做空", "下方做多", "下方做空", "區內做多", "區內做空"])}
def stat(x, tp=55):
    p = x[f"pnl{tp}"]; return len(p), (p > 0).mean() * 100 if len(p) else np.nan, p.mean() if len(p) else np.nan
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
# A. 基準：每條規則
print("=== A（略）")
A = []
for k, x in F.groupby("key"):
    a = stat(x[x.W == "REC"]); b = stat(x[x.W == "HIST"])
    A.append(dict(key=k, rec_n=a[0], rec_win=a[1], rec_avg=a[2], hist_n=b[0], hist_win=b[1], hist_avg=b[2]))
A = pd.DataFrame(A).round(1)
# B. 單一條件 × 規則
B = []
for k, x in F.groupby("key"):
    for fname, (col, vals) in FILT.items():
        for v in vals:
            y = x[x[col] == v]
            a = stat(y[y.W == "REC"]); b = stat(y[y.W == "HIST"])
            B.append(dict(key=k, filt=f"{fname}={v}", rec_n=a[0], rec_win=a[1], rec_avg=a[2], hist_n=b[0], hist_win=b[1], hist_avg=b[2]))
B = pd.DataFrame(B)
ok = B[(B.rec_n >= 4) & (B.hist_n >= 20) & (B.rec_avg > 0) & (B.hist_avg > 0)].copy()
ok["min_win"] = ok[["rec_win", "hist_win"]].min(axis=1)
print("\n=== B. 規則＋一個條件：兩段都賺、按兩段較低的勝率排序（前 25）")
print(ok.sort_values(["min_win", "hist_n"], ascending=False).head(25).round(1).to_string(index=False))
# C. 條件本身（所有規則合併）能不能提高勝率
print("\n=== C. 條件對所有規則合併的勝率影響（止盈 55）")
C = []
for fname, (col, vals) in FILT.items():
    for v in vals:
        for W in ["HIST", "REC"]:
            x = F[F.W == W]; y = x[x[col] == v]; z = x[x[col] != v]
            C.append(dict(filt=f"{fname}={v}", W=W, n=len(y), win=(y.pnl55 > 0).mean() * 100, others=(z.pnl55 > 0).mean() * 100, avg=y.pnl55.mean()))
C = pd.DataFrame(C); C["lift"] = C.win - C.others
print(C.pivot(index="filt", columns="W", values=["n", "win", "lift", "avg"]).round(1).to_string())
B.to_csv("wr_single_filters.csv", index=False); A.to_csv("wr_rules.csv", index=False)
