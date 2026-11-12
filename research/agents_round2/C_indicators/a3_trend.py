"""(2) 趨勢日判斷器 → 直接順勢單。研究用途，不是投資建議。
判斷器 TDv（投票）：10:00（或 10:30）時 5 個盤中動能指標
  effi（開盤以來方向效率＝淨移動/總路徑，帶號）、above_vw（在 VWAP 上方的時間比例）、
  kema_slope（5 分 EMA20 近 30 分斜率/ATR）、kdi（5 分 DI+−DI−）、kmacdh（5 分 MACD 柱/ATR）
各自「強」＝|值| > 探索段分位數 q；強票數（同方向）≥ k 且沒有反方向強票 → 趨勢日，方向＝票的方向。
否則＝非趨勢日（盤整日）。
直接單：決策時間那一分鐘開盤進場（訊號只用之前收盤的 K 棒），15:55 出場，停損 SL，止盈 TP（None＝不設）。
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/wt/research")
import numpy as np, pandas as pd
from lib import data, sim
import cbook as B
g = data.load_grid("NQ", include_holdout=True)
D = pd.read_pickle(os.path.join(HERE, "days2.pkl"))
D["i"] = np.arange(len(D))
OK = (D.tradable & D.has_prev).values
VOTE = ["effi", "above_vw", "kema_slope", "kdi", "kmacdh"]
CEN = {}
SLOT = {"10": 960, "1030": 990}


def td_signal(tag="10", q=0.67, k=3, feats=VOTE):
    disc = (D.P == "探索").values & OK
    up = np.zeros(len(D)); dn = np.zeros(len(D))
    for f in feats:
        x = D[f"{f}_{tag}"].values - CEN.get(f, 0)
        thr = np.nanquantile(np.abs(x[disc]), q)
        up += np.nan_to_num(x) > thr; dn += np.nan_to_num(x) < -thr
    d = np.where((up >= k) & (dn == 0), 1, np.where((dn >= k) & (up == 0), -1, 0))
    return np.where(OK, d, 0)


def run_direct(dirv, tag="10", sl=145.0, tp=None, trail=None):
    S = [dict(i=int(i), entry_slot=SLOT[tag], dir=int(dirv[i]), exit_slot=930 + 385) for i in np.where(dirv != 0)[0]]
    kw = dict(tp_pts=tp if tp else np.inf, sl_pts=sl)
    if trail: kw.update(trail_pts=trail, trail_trigger=trail)
    T = sim.run_signals(g, S, **kw)
    T["P"] = B.period(T.date) if hasattr(B, "period") else None
    return T


def period(d):
    d = np.asarray(d, dtype=str)
    return np.select([d <= "2025-06-30", d <= "2025-12-31", d <= "2026-09-30"], ["探索", "驗證", "測試"], "近期")


def summ(T, worst):
    T = T.copy(); T["P"] = period(T.date); T["worst"] = worst
    out = {}
    for p in B.PER + ["全期"]:
        x = T if p == "全期" else T[T.P == p]
        dy = x.groupby("date").pnl.sum().sort_index()
        out[p] = dict(n=len(x), win=round(100 * (x.pnl > 0).mean(), 1) if len(x) else np.nan, avg=round(x.pnl.mean()) if len(x) else np.nan,
                      tot=round(x.pnl.sum()), dd=round(B.dd(dy.values)), worst=round(dy.min()) if len(dy) else np.nan)
    return out, T


if __name__ == "__main__":
    rows = []; nvar = 0
    for tag in ["10", "1030"]:
        for q in [0.5, 0.67, 0.8]:
            for k in [2, 3, 4, 5]:
                dv = td_signal(tag, q, k)
                for sl, tp, trail in [(145, None, None), (100, None, None), (145, 150, None), (145, None, 100)]:
                    T = run_direct(dv, tag, sl, tp, trail); nvar += 1
                    s, _ = summ(T, sl * 20 + 48)
                    r = dict(time=tag, q=q, k=k, sl=sl, tp=tp, trail=trail)
                    for p in B.PER + ["全期"]:
                        for m in ["n", "win", "avg", "tot", "dd"]:
                            r[f"{p}_{m}"] = s[p][m]
                    rows.append(r)
    R = pd.DataFrame(rows)
    R["pass3"] = (R.探索_avg > 0) & (R.驗證_avg > 0) & (R.測試_avg > 0)
    R.to_csv(os.path.join(HERE, "out_a3_direct.csv"), index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
    print("直接順勢單變體數", nvar, "；三段都賺", int(R.pass3.sum()))
    cols = ["time", "q", "k", "sl", "tp", "trail"] + [f"{p}_{m}" for p in B.PER for m in ["n", "avg", "tot"]] + ["全期_tot", "全期_dd"]
    print(R[cols].to_string(index=False))
