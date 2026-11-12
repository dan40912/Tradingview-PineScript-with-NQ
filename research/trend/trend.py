"""局勢判斷與順勢進場（research/trend/PREREG.md）。研究用途，不是投資建議。"""
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib，以及 research/winrate_recent/wr_table.csv.gz；放到該分支 research/ 下執行（見 docs/23）。
import sys, os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from lib import data, features, sim
from lib.data import RTH_OFF as R0
g = data.load_grid("NQ", include_holdout=True); lv = features.daily_levels(g); L = lv.set_index("i")
# 日線趨勢 D（只用前一天以前的收盤）
C = g.rth(g.C); last = np.array([r[~np.isnan(r)][-1] if np.isfinite(r).any() else np.nan for r in C])
cl = pd.Series(last, index=g.dates)[g.has_rth]
sma = cl.rolling(20).mean(); slope = sma - sma.shift(5)
Draw = np.where((cl > sma) & (slope > 0), 1, np.where((cl < sma) & (slope < 0), -1, 0))
Dser = pd.Series(Draw, index=cl.index).shift(1).reindex(g.dates).ffill().fillna(0).astype(int)   # 用前一天的狀態
# 盤中 VWAP（09:30 起，到每分鐘為止）
def vwap_row(i):
    H, Lw, Cc, V = g.H[i][R0:R0 + 390], g.L[i][R0:R0 + 390], g.C[i][R0:R0 + 390], g.V[i][R0:R0 + 390]
    tp = (H + Lw + Cc) / 3; m = np.isfinite(tp) & np.isfinite(V)
    pv = np.cumsum(np.where(m, tp * V, 0)); vv = np.cumsum(np.where(m, V, 0))
    return np.where(vv > 0, pv / np.where(vv > 0, vv, 1), np.nan)
def signals():
    out = []
    for r in lv.itertuples():
        if not r.has_prev or not g.tradable[r.i] or np.isnan(r.open_0930): continue
        i = r.i; H, Lw, Cc, O = g.H[i], g.L[i], g.C[i], g.O[i]; vw = vwap_row(i); D = Dser[g.dates[i]]
        orh = np.nanmax(H[R0:R0 + 30]); orl = np.nanmin(Lw[R0:R0 + 30])
        done = set(); below_pdh = O[R0] < r.pdh; above_pdl = O[R0] > r.pdl
        prev_c = None; cntH = cntL = 0
        for idx in range(4, 120, 5):
            s = R0 + idx; c = Cc[s]
            if np.isnan(c): continue
            bh = np.nanmax(H[s - 4:s + 1]); bl = np.nanmin(Lw[s - 4:s + 1]); v = vw[idx]
            ei = idx + 1; ok = not (28 <= ei <= 32) and ei <= 120
            if bl < r.pdh: below_pdh = True
            if bh > r.pdl: above_pdl = True
            def add(rule, d):
                if (rule, d) in done or not ok: return
                done.add((rule, d)); out.append(dict(i=i, entry_slot=R0 + ei, dir=d, exit_slot=R0 + 179, rule=rule, D=D))
            # T1 VWAP 回檔
            if idx >= 14 and prev_c is not None and not np.isnan(v):
                if prev_c > vw[idx - 5] and bl <= v and c > v: add("T1", 1)
                if prev_c < vw[idx - 5] and bh >= v and c < v: add("T1", -1)
            # T2 前日高低突破（連續 2 根）
            cntH = cntH + 1 if (c > r.pdh and below_pdh) else 0
            cntL = cntL + 1 if (c < r.pdl and above_pdl) else 0
            if cntH >= 2: add("T2", 1)
            if cntL >= 2: add("T2", -1)
            # T3 開盤區間突破（10:00 後第一根）
            if idx >= 34:
                if c > orh: add("T3", 1)
                if c < orl: add("T3", -1)
            prev_c = c
    return out
S = signals()
T = sim.run_signals(g, S, tp_pts=55.0, sl_pts=145.0)
T["aligned"] = T.dir == T.D
T["P"] = np.select([T.date >= "2026-10-01", T.date >= "2026-01-01", T.date >= "2025-01-01"], ["近期", "2026-1~9", "2025"], "2024")
T.to_pickle("trend_trades.pkl")
pd.set_option("display.width", 250)
def show(x, name):
    s = x.groupby("P").pnl.agg(["size", lambda p: (p > 0).mean() * 100, "mean"]).round(0)
    s.columns = ["n", "win", "avg"]; s = s.reindex(["2024", "2025", "2026-1~9", "近期"])
    h = x[x.P != "近期"].pnl
    print(f"{name:26s} 過去 {len(h):3d} 筆 勝率 {(h>0).mean()*100:4.1f}% 每筆 {h.mean():+5.0f}｜" + "｜".join(f"{p}:{int(r.n) if r.n==r.n else 0}筆 {r.win:.0f}% {r.avg:+.0f}" for p, r in s.iterrows()))
for rule in ["T1", "T2", "T3"]:
    for side, sd in [("多", 1), ("空", -1)]:
        x = T[(T.rule == rule) & (T.dir == sd)]
        show(x, f"{rule}{side} 不分局勢"); show(x[x.aligned], f"{rule}{side} 順日線趨勢"); show(x[(x.D == 0)], f"{rule}{side} 盤整日")
