"""補充測試。研究用途，不是投資建議。
(a) 10:00 判斷器「逆向」時，提早在 10:00 出清 10:00 前進場、還沒出場的基準單（只用 10:00 前資訊做決定）。
(b) 缺口守住（為了 11/11 這種日子）：開在前高之上（或前低之下），到決策時間（10:00／10:30）為止
    盤中低點沒有跌回前高下（多）／高點沒有回到前低上（空）→ 決策時間順缺口方向進場，停損 SL，15:55 出場。
    守住的線：前高/前低（PDH）或缺口一半（(開盤+前收)/2）。
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import a4_alloc as AL
import a3_trend as A
import cbook as B
from lib import sim
from lib.data import RTH_OFF as R0
D, g = A.D, A.g
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
COLS = ["date", "P", "pt", "dir", "entry_slot", "exit_slot", "pnl", "worst"]


def early_exit(dv, cut=960):
    """回傳一個 drop_fn：10:00 前進場、判斷器逆向的基準單，改成 cut-1 收盤出場（重算）。"""
    dmap = dict(zip(g.dates, dv))
    def f(T):
        T = T.copy()
        d = T.date.map(dmap).fillna(0).values
        m = (T.pt != "TD順勢") & (T.entry_slot < cut) & (T.exit_slot >= cut) & (d != 0) & (T.dir.values == -d)
        for k in np.where(m)[0]:
            r = T.iloc[k]
            x = sim.simulate_one(g, g.dindex[r.date], int(r.entry_slot), int(r.dir), tp_pts=55.0, sl_pts=145.0, exit_slot=cut - 1)
            T.iloc[k, T.columns.get_loc("pnl")] = x["pnl"]; T.iloc[k, T.columns.get_loc("exit_slot")] = x["exit_slot"]
        return T
    return f


def gap_hold(tag="1030", line="pdh", sl=100.0, sides=(1, -1)):
    t = A.SLOT[tag]
    S = []
    for r in D.itertuples():
        i = r.i
        if not A.OK[i] or not np.isfinite(r.open_0930) or not np.isfinite(r.pdh):
            continue
        lo = np.nanmin(g.L[i][R0:t]); hi = np.nanmax(g.H[i][R0:t])
        if 1 in sides and r.open_0930 > r.pdh:
            lvl = r.pdh if line == "pdh" else (r.open_0930 + r.pdc) / 2
            if lo > lvl: S.append(dict(i=i, entry_slot=t, dir=1, exit_slot=R0 + 385))
        if -1 in sides and r.open_0930 < r.pdl:
            lvl = r.pdl if line == "pdh" else (r.open_0930 + r.pdc) / 2
            if hi < lvl: S.append(dict(i=i, entry_slot=t, dir=-1, exit_slot=R0 + 385))
    T = sim.run_signals(g, S, tp_pts=np.inf, sl_pts=sl)
    T["P"] = A.period(T.date); T["pt"] = "缺口守住"; T["worst"] = sl * 20 + 48
    return T


if __name__ == "__main__":
    print("== (a) 判斷器逆向就 10:00 出清")
    for q, k in [(0.67, 3), (0.8, 3), (0.67, 4), (0.8, 4)]:
        dv = A.td_signal("10", q, k)
        TD = AL.td_trades(dv, 100.0)
        for name, T in [("基準", B.book()), ("基準＋提早出清", B.book(drop_fn=early_exit(dv))),
                        ("A 基準＋TD單", B.book(extra=TD)), ("F A＋提早出清", B.book(extra=TD, drop_fn=early_exit(dv)))]:
            b = B.brief(T)
            print(f"  q{q} k{k} {name:14s}", "｜".join(f"{p} {b[p][0]:+8,.0f} DD {b[p][1]:+7,.0f}" for p in B.PER + ["全期"]))
    print("\n== (b) 缺口守住（直接單，1 口）")
    rows = []
    for tag in ["10", "1030"]:
        for line in ["pdh", "half"]:
            for sl in [100.0, 145.0]:
                for sides, sn in [((1,), "只多"), ((1, -1), "多空")]:
                    T = gap_hold(tag, line, sl, sides)
                    r = dict(time=tag, line=line, sl=sl, side=sn)
                    for p in B.PER:
                        x = T[T.P == p]; r[f"{p}_n"] = len(x); r[f"{p}_avg"] = round(x.pnl.mean()) if len(x) else np.nan
                    r["全期_tot"] = round(T.pnl.sum()); r["全期_dd"] = round(B.dd(T.sort_values("date").pnl.values))
                    r["11/11"] = T[T.date == "2026-11-11"].pnl.sum()
                    rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "out_a6_gaphold.csv"), index=False)
    print(R.to_string(index=False))
