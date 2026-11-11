"""第三批的組合：I2＋通過的點，每筆 5 口 MNQ（0.5 口 NQ），每日停損 −$2,900。研究用途，不是投資建議。"""
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib；放到該分支 research/ 下執行（見 docs/21 第 9 節）。
import sys, os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import f_family as FF
from lib import sim, strategies
from lib.data import RTH_OFF as R0
g, lv = FF.NQ.g, FF.NQ.lv
SIG = {"I2": [dict(s, exit_slot=R0 + 179) for s in strategies.i2_v1_signals(g, lv)],
       "前高空": FF.build("前高空", False, False), "前高空ES": FF.build("前高空", True, False),
       "亞洲多ES": FF.build("亞洲多", True, False), "前低多": FF.build("前低多", False, False),
       "亞洲空": FF.build("亞洲空", False, False), "亞洲多": FF.build("亞洲多", False, False)}
def book(names, qty=0.5, daystop=2900.0):
    T = pd.concat([sim.run_signals(g, SIG[n], tp_pts=55.0, sl_pts=145.0, contracts=qty).assign(name=n) for n in names])
    T = T.sort_values(["date", "entry_slot"]); keep = []
    for d, x in T.groupby("date", sort=False):
        closed = []
        for t in x.itertuples():
            if sum(p for xs, p in closed if xs <= t.entry_slot) <= -daystop: keep.append(False); continue
            keep.append(True); closed.append((t.exit_slot, t.pnl))
    T["taken"] = keep; return T[T.taken]
sp = lambda d: np.select([d <= "2025-06-30", d <= "2025-12-31", d <= "2026-10-19"], ["1 探索", "2 驗證", "3 測試"], "4 前進")
def dd(x): c = x.cumsum(); return (c - c.cummax()).min()
BOOKS = {"A 只做 I2（10 口）": (["I2"], 1.0), "B I2＋前高空": (["I2", "前高空"], 0.5),
         "H I2＋前高空＋亞洲多ES": (["I2", "前高空", "亞洲多ES"], 0.5),
         "H' I2＋前高空ES＋亞洲多ES（照挑選規則）": (["I2", "前高空ES", "亞洲多ES"], 0.5),
         "D 五個點（未過濾）": (["I2", "前高空", "前低多", "亞洲空", "亞洲多"], 0.5)}
if __name__ == "__main__":
    days = pd.Series([d for d, t, h in zip(g.dates, g.tradable, lv.has_prev) if t and h]); dsp = sp(days.values)
    out = []; allT = []
    for k, (names, q) in BOOKS.items():
        T = book(names, q); T["sp"] = sp(T.date.values); allT.append(T.assign(book=k))
        for s, x in T.groupby("sp"):
            out.append(dict(book=k, sp=s, n=len(x), per_day=round(len(x) / (dsp == s).sum(), 2), win=round((x.pnl > 0).mean() * 100, 1),
                            avg=round(x.pnl.mean()), total=round(x.pnl.sum()), maxDD=round(dd(x.pnl)), ret_dd=round(x.pnl.sum() / -dd(x.pnl), 2)))
    R = pd.DataFrame(out); R.to_csv("f_book_summary.csv", index=False); pd.concat(allT).to_csv("f_book_trades.csv", index=False)
    pd.set_option("display.width", 250)
    for c in ["total", "maxDD", "ret_dd", "win", "per_day"]:
        print("=====", c); print(R.pivot(index="book", columns="sp", values=c).reindex(list(BOOKS)).to_string())
