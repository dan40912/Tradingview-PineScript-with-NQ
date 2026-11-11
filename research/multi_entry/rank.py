"""用使用者的預算（MNQ、每天最多虧 $3,000、+55／−145）排出全期間總獲利最高、勝率最高的組合。研究用途，不是投資建議。"""
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib；放到該分支 research/ 下執行（見 docs/21 第 10 節）。
import sys, os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import f_family as FF
from lib import sim, strategies
from lib.data import RTH_OFF as R0
g, lv = FF.NQ.g, FF.NQ.lv
SIG = {"I2": [dict(s, exit_slot=R0 + 179) for s in strategies.i2_v1_signals(g, lv)],
       "前高空": FF.build("前高空", False, False), "前低多": FF.build("前低多", False, False),
       "亞洲空": FF.build("亞洲空", False, False), "亞洲多": FF.build("亞洲多", False, False),
       "亞洲多ES": FF.build("亞洲多", True, False), "前低多ES順I2": FF.build("前低多", True, True)}
PT, COST = 2.0, 4.8
def trade(s, plan, loss, target):
    i, d, e, xs = s["i"], s["dir"], s["entry_slot"], s["exit_slot"]
    O, H, L, C = g.O[i], g.H[i], g.L[i], g.C[i]
    while e <= xs and np.isnan(O[e]): e += 1
    if e > xs: return None
    p0 = O[e]; pos = [(p0, q) for q, a in plan if a is None]; pend = [(p0 - d * a, q) for q, a in plan if a is not None]
    pnl = lambda p: sum(q * (p - a) * d for a, q in pos) * PT
    for t in range(e, xs + 1):
        h, l = H[t], L[t]
        if np.isnan(h): continue
        w, b = (l, h) if d == 1 else (h, l); added = False
        for p, q in list(pend):
            if (d == 1 and w <= p) or (d == -1 and w >= p): pos.append((p, q)); pend.remove((p, q)); added = True
        if pnl(w) <= -loss: return t, -loss - COST * sum(q for _, q in pos)
        if not added and pnl(b) >= target: return t, target - COST * sum(q for _, q in pos)
    cc = C[e:xs + 1]; last = cc[~np.isnan(cc)][-1]
    return xs, pnl(last) - COST * sum(q for _, q in pos)
def book(names, plan, loss, target, daystop=3000.0):
    rows = []
    for n in names:
        for s in SIG[n]:
            r = trade(s, plan, loss, target)
            if r: rows.append(dict(date=g.dates[s["i"]], name=n, es=s["entry_slot"], xs=r[0], net=r[1]))
    T = pd.DataFrame(rows).sort_values(["date", "es"]); keep = []
    for d, x in T.groupby("date", sort=False):
        closed = []
        for t in x.itertuples():
            if sum(p for xs, p in closed if xs <= t.es) <= -daystop + 100: keep.append(False); continue
            keep.append(True); closed.append((t.xs, t.net))
    T["taken"] = keep; return T[T.taken]
sp = lambda d: np.select([d <= "2025-06-30", d <= "2025-12-31", d <= "2026-10-19"], ["探索", "驗證", "測試"], "前進")
def dd(x): c = x.cumsum(); return (c - c.cummax()).min()
ALL5 = ["I2", "前高空", "前低多", "亞洲空", "亞洲多"]
ALL5ES = ["I2", "前高空", "前低多", "亞洲空", "亞洲多ES"]
H = ["I2", "前高空", "亞洲多ES"]
P10, P5, DCA = [(10, None)], [(5, None)], [(5, None), (5, 75)]
BOOKS = {
 "只做 I2｜10 口": (["I2"], P10, 2900, 1100),
 "H｜各 5 口": (H, P5, 1450, 550), "H｜各 10 口": (H, P10, 2900, 1100), "H｜DCA 5+5": (H, DCA, 3000, 1100),
 "五個點｜各 5 口": (ALL5, P5, 1450, 550), "五個點｜各 10 口": (ALL5, P10, 2900, 1100), "五個點｜DCA 5+5": (ALL5, DCA, 3000, 1100),
 "五個點(亞洲多改ES)｜各 5 口": (ALL5ES, P5, 1450, 550), "五個點(亞洲多改ES)｜各 10 口": (ALL5ES, P10, 2900, 1100),
 "五個點(亞洲多改ES)｜DCA 5+5": (ALL5ES, DCA, 3000, 1100),
 "I2＋前高空＋亞洲多ES＋亞洲空｜各 10 口": (["I2", "前高空", "亞洲多ES", "亞洲空"], P10, 2900, 1100),
}
if __name__ == "__main__":
    rows = []
    for k, (names, plan, loss, tgt) in BOOKS.items():
        T = book(names, plan, loss, tgt); T["sp"] = sp(T.date.values)
        r = dict(book=k, n=len(T), win=round((T.net > 0).mean() * 100, 1), total=round(T.net.sum()), maxDD=round(dd(T.net)),
                 worst_day=round(T.groupby("date").net.sum().min()))
        for s in ["探索", "驗證", "測試", "前進"]:
            x = T[T.sp == s]; r[s] = round(x.net.sum()); r[s + "勝率"] = round((x.net > 0).mean() * 100, 1)
        rows.append(r)
    R = pd.DataFrame(rows).sort_values("total", ascending=False); R.to_csv("rank_summary.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(R.to_string(index=False))
