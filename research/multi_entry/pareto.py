"""所有已定義的點×過濾的組合（450 種），每筆 10 口 MNQ、每日停損 −$2,900，找總獲利和勝率都最高的組合。研究用途，不是投資建議。"""
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib；放到該分支 research/ 下執行（見 docs/21 第 10 節）。
import sys, os, itertools; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import f_family as FF
from lib import sim, strategies
from lib.data import RTH_OFF as R0
g, lv = FF.NQ.g, FF.NQ.lv
def run(sigs, name):
    T = sim.run_signals(g, sigs, tp_pts=55.0, sl_pts=145.0); T["name"] = name; return T[["date", "entry_slot", "exit_slot", "pnl", "name"]]
V = {"I2": run([dict(s, exit_slot=R0 + 179) for s in strategies.i2_v1_signals(g, lv)], "I2")}
for p in ["前高空", "前低多"]:
    for f1, f2, tag in [(False, False, ""), (True, False, "ES"), (False, True, "順I2"), (True, True, "ES順I2")]:
        V[p + tag] = run(FF.build(p, f1, f2), p + tag)
for p in ["亞洲空", "亞洲多"]:
    for f1, tag in [(False, ""), (True, "ES")]:
        V[p + tag] = run(FF.build(p, f1, False), p + tag)
sp = lambda d: np.select([d <= "2025-06-30", d <= "2025-12-31", d <= "2026-10-19"], ["探索", "驗證", "測試"], "前進")
def book(names):
    T = pd.concat([V[n] for n in names]).sort_values(["date", "entry_slot"]); keep = []
    for d, x in T.groupby("date", sort=False):
        closed = []
        for t in x.itertuples():
            if sum(p for xs, p in closed if xs <= t.entry_slot) <= -2900: keep.append(False); continue
            keep.append(True); closed.append((t.exit_slot, t.pnl))
    return T[keep]
def dd(x): c = x.cumsum(); return (c - c.cummax()).min()
OPT = {"I2": [None, "I2"], "前高空": [None, "前高空", "前高空ES", "前高空順I2", "前高空ES順I2"],
       "前低多": [None, "前低多", "前低多ES", "前低多順I2", "前低多ES順I2"], "亞洲空": [None, "亞洲空", "亞洲空ES"], "亞洲多": [None, "亞洲多", "亞洲多ES"]}
rows = []
for combo in itertools.product(*OPT.values()):
    names = [c for c in combo if c]
    if not names: continue
    T = book(names); per = T.groupby(sp(T.date.values)).pnl.sum()
    rows.append(dict(book="＋".join(names), k=len(names), n=len(T), win=round((T.pnl > 0).mean() * 100, 2), total=round(T.pnl.sum()),
                     maxDD=round(dd(T.pnl)), **{s: round(per.get(s, 0)) for s in ["探索", "驗證", "測試", "前進"]}))
R = pd.DataFrame(rows); R["all4"] = (R[["探索", "驗證", "測試", "前進"]] > 0).all(axis=1)
R.to_csv("pareto_all.csv", index=False)
Q = R[R.all4].copy()
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 60)
print("books:", len(R), "positive in all 4 periods:", len(Q))
print("== top 12 by total (all4)"); print(Q.sort_values("total", ascending=False).head(12).to_string(index=False))
print("== top 12 by win (all4)"); print(Q.sort_values(["win", "total"], ascending=False).head(12).to_string(index=False))
# Pareto frontier
Q = Q.sort_values(["total", "win"], ascending=False); front = []; best = -1
for r in Q.itertuples():
    if r.win > best: front.append(r.book); best = r.win
print("== Pareto frontier (total desc, win asc):"); print(Q[Q.book.isin(front)].to_string(index=False))
