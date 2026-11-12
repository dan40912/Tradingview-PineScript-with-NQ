"""最終候選四段表、年度表、穩健性。研究用途，不是投資建議。輸出 out_final.md。"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import a4_alloc as AL
import a3_trend as A
import a6_more as M
import cbook as B
from lib import sim
g, D = A.g, A.D
out = []
dv8 = A.td_signal("10", 0.8, 3); dv67 = A.td_signal("10", 0.67, 3)
TD8 = AL.td_trades(dv8, 100.0); TD67 = AL.td_trades(dv67, 100.0)
BOOKS = {"基準 順勢多點": B.book(),
         "候選1 基準＋TD10單(q80,k3,停損100)": B.book(extra=TD8),
         "候選2 基準＋TD10單(q67,k3,停損100)＋逆向提早出清": B.book(extra=TD67, drop_fn=M.early_exit(dv67))}
allr = pd.concat([B.stats(T, k) for k, T in BOOKS.items()])
out.append("## 四段績效（1 口 NQ，扣成本，每日 $3,000 風控）\n\n" + B.md(allr))
# 年度
def yr(T):
    y = np.select([T.date >= "2026-10-01", T.date >= "2026-01-01", T.date >= "2025-01-01"], ["近期", "2026-1~9", "2025"], "2024")
    return T.assign(Y=y).groupby("Y").pnl.sum().reindex(["2024", "2025", "2026-1~9", "近期"]).round(0)
yt = pd.DataFrame({k: yr(T) for k, T in BOOKS.items()}).T
out.append("## 年度合計\n\n" + yt.to_markdown())
# TD 單本身
for nm, T in [("TD10 q80 k3", TD8), ("TD10 q67 k3", TD67)]:
    s = B.stats(B.risk(T[["date", "P", "pt", "dir", "entry_slot", "exit_slot", "pnl", "worst"]]), nm + " 單獨")
    out.append(f"## {nm} 單獨（10:00 進、停損 100、15:55 出）\n\n" + B.md(s))
    p = T.pnl.values; tt = p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))
    lines = [f"- 全期 t 值 {tt:.2f}；做多 {int((T.dir == 1).sum())} 筆每筆 {T[T.dir == 1].pnl.mean():+.0f}、做空 {int((T.dir == -1).sum())} 筆每筆 {T[T.dir == -1].pnl.mean():+.0f}；出場原因 {T.reason.value_counts().to_dict()}"]
    for per in B.PER:
        x = T[T.P == per]
        lines.append(f"- {per}：多 {int((x.dir == 1).sum())} 筆 {x[x.dir == 1].pnl.mean():+.0f}／空 {int((x.dir == -1).sum())} 筆 {x[x.dir == -1].pnl.mean():+.0f}")
    out.append("\n".join(lines))
# 穩健性：晚 1/5 分鐘進場、15:30 出場、成本加倍、停損 80/120
rob = []
for nm, kw in [("原版", dict()), ("晚 1 分進場", dict(delay=1)), ("晚 5 分進場", dict(delay=5)), ("15:30 出場", dict(exit=930 + 360)),
               ("停損 80", dict(sl=80.0)), ("停損 120", dict(sl=120.0)), ("成本加倍", dict(cost=2))]:
    for qn, dv in [("q80k3", dv8), ("q67k3", dv67)]:
        S = [dict(i=int(i), entry_slot=960 + kw.get("delay", 0), dir=int(dv[i]), exit_slot=kw.get("exit", 930 + 385)) for i in np.where(dv != 0)[0]]
        T = sim.run_signals(g, S, tp_pts=np.inf, sl_pts=kw.get("sl", 100.0))
        if kw.get("cost"): T["pnl"] = T.pnl - 48
        T["P"] = A.period(T.date)
        r = dict(變化=nm, 判斷器=qn)
        for per in B.PER:
            r[per] = round(T[T.P == per].pnl.mean())
        r["全期合計"] = round(T.pnl.sum())
        rob.append(r)
out.append("## TD10 單穩健性（每筆平均）\n\n" + pd.DataFrame(rob).to_markdown(index=False))
open(os.path.join(HERE, "out_final.md"), "w").write("\n\n".join(out))
print("\n\n".join(out))
for k, T in BOOKS.items():
    T.to_csv(os.path.join(HERE, f"trades_{k.split()[0]}.csv"), index=False)
