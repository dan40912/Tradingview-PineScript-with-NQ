"""組合：I2＋通過或待驗證的參考點，MNQ 計價、每日停損、DCA 比較。研究用途，不是投資建議。"""
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib；放到該分支 research/ 下執行（見 docs/21 第 10 節）。
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from lib import data, features, strategies
from lib.data import RTH_OFF as R0
import family as F
g, lv = F.g, F.lv; LVI = lv.set_index("i")
fam = pd.DataFrame(F.signals())
def pick(rule, side): return [dict(r) for r in fam[(fam.rule == rule) & (fam.side == side)].to_dict("records")]
SIG = {"I2": [dict(s, exit_slot=R0 + 179) for s in strategies.i2_v1_signals(g, lv)],
       "前高空": pick("AM-PD-k1", "空"), "前低多": pick("AM-PD-k1", "多"),
       "亞洲空": pick("ASIA-PD-k1", "空"), "亞洲多": pick("ASIA-PD-k1", "多")}
PT, COST = 2.0, 4.8
def trade(s, plan, loss=3000.0, target=1100.0):
    """plan: [(qty, adverse_pts or None)] ; whole position exits at +target / -loss (dollars) or time."""
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
        if pnl(w) <= -loss: return dict(exit_slot=t, gross=-loss, qty=sum(q for _, q in pos), reason="sl")
        if not added and pnl(b) >= target: return dict(exit_slot=t, gross=target, qty=sum(q for _, q in pos), reason="tp")
    cc = C[e:xs + 1]; last = cc[~np.isnan(cc)][-1]
    return dict(exit_slot=xs, gross=pnl(last), qty=sum(q for _, q in pos), reason="time")
def book(names, plan, loss, target, daystop=2900.0):
    rows = []
    for n in names:
        for s in SIG[n]:
            r = trade(s, plan, loss, target)
            if r: rows.append(dict(date=g.dates[s["i"]], name=n, entry_slot=s["entry_slot"], exit_slot=r["exit_slot"],
                                   net=r["gross"] - COST * r["qty"], reason=r["reason"]))
    T = pd.DataFrame(rows).sort_values(["date", "entry_slot"])
    keep = []
    for d, x in T.groupby("date", sort=False):
        closed = []
        for t in x.itertuples():
            if sum(p for xs, p in closed if xs <= t.entry_slot) <= -daystop: keep.append(False); continue
            keep.append(True); closed.append((t.exit_slot, t.net))
    T["taken"] = keep; return T[T.taken]
sp = lambda d: np.select([d <= "2025-06-30", d <= "2025-12-31", d <= "2026-10-19"], ["1 探索", "2 驗證", "3 測試"], "4 前進")
def dd(x): c = x.cumsum(); return (c - c.cummax()).min()
BOOKS = {
 "A 只做 I2，10 口": (["I2"], [(10, None)], 2900, 1100),
 "B I2＋前高空，各 5 口": (["I2", "前高空"], [(5, None)], 1450, 550),
 "C I2＋前高空，各 10 口": (["I2", "前高空"], [(10, None)], 2900, 1100),
 "D 五個點，各 5 口": (["I2", "前高空", "前低多", "亞洲空", "亞洲多"], [(5, None)], 1450, 550),
 "E 五個點，各 10 口": (["I2", "前高空", "前低多", "亞洲空", "亞洲多"], [(10, None)], 2900, 1100),
 "F 五個點，DCA 5+5（逆向 75 點）": (["I2", "前高空", "前低多", "亞洲空", "亞洲多"], [(5, None), (5, 75)], 3000, 1100),
 "G I2＋前高空，DCA 5+5（逆向 75 點）": (["I2", "前高空"], [(5, None), (5, 75)], 3000, 1100),
}
if __name__ == "__main__":
    out = []; tr = {}
    days = pd.Series([d for d, t, h in zip(g.dates, g.tradable, lv.has_prev) if t and h]); dsp = sp(days.values)
    for k, (names, plan, loss, tgt) in BOOKS.items():
        T = book(names, plan, loss, tgt); T["sp"] = sp(T.date.values); tr[k] = T
        for s, x in T.groupby("sp"):
            dy = x.groupby("date").net.sum(); nd = (dsp == s).sum()
            out.append(dict(book=k, sp=s, n=len(x), per_day=round(len(x) / nd, 2), win=round((x.net > 0).mean() * 100, 1),
                            avg=round(x.net.mean()), total=round(x.net.sum()), maxDD=round(dd(x.net)), worst_day=round(dy.min()),
                            ret_dd=round(x.net.sum() / -dd(x.net), 2) if dd(x.net) < 0 else np.nan))
    R = pd.DataFrame(out); R.to_csv("combo_summary.csv", index=False)
    pd.concat([t.assign(book=k) for k, t in tr.items()]).to_csv("combo_trades.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    for c in ["total", "maxDD", "ret_dd", "win", "per_day"]:
        print("=====", c); print(R.pivot(index="book", columns="sp", values=c).reindex(list(BOOKS)).to_string())
