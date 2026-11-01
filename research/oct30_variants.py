"""10/30 這種日子，順勢規則的幾個變化：當天結果＋歷史（2024-01～2026-09）＋10 月全月。需要 build_cache.py --yahoo。"""
import pickle, numpy as np, pandas as pd, strat, winrate_opt as W
from eng import sim
nq = pickle.load(open("NQ_days2.pkl", "rb")); hh = lambda i: f"{9 + (30 + i) // 60:02d}:{(30 + i) % 60:02d}"
def reentry(d, base=dict(ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="outpd", be=1.0), sl=60, rr=2.0, last=210, xi=379):
    """（10/30 之後才想到的假設）順勢第一筆止損後，5 分K 收盤創當天新低（新高）→ 同方向再進一次。"""
    t = strat.trend(d, **base)
    if not t or t["res"] != "SL": return t and None
    s = t["side"]; C, H, L = d["C"], d["H"], d["L"]
    for i in range(t["xi"] // 5 * 5 + 4, last, 5):
        if np.isnan(C[i]): continue
        ext = np.nanmin(L[:i - 4]) if s == -1 else np.nanmax(H[:i - 4])
        if (C[i] < ext) if s == -1 else (C[i] > ext):
            r = sim(d, i + 1, s, rr * sl, sl, xi, None, None, "順勢再進場")
            return r
    return None
V = {"D-2 順勢（IB30、SL 30–80、2R、12:29）": lambda d: strat.trend(d, ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="outpd", be=1.0),
     "停損改 ATR×2.5": lambda d: W.trendx(d, rr=2.0, stop="atr2.5", xi=179, be=1.0),
     "IB 60 分、SL 20–60、2R、15:49（組合 E）": lambda d: strat.trend(d, ib=59, smin=20, smax=60, rr=2.0, xi=379, cond="outpd"),
     "IB 60 分、SL 30–80、2R、15:49": lambda d: strat.trend(d, ib=59, smin=30, smax=80, rr=2.0, xi=379, cond="outpd"),
     "SL 30–80、2R、15:49 出場": lambda d: strat.trend(d, ib=29, smin=30, smax=80, rr=2.0, xi=379, cond="outpd", be=1.0),
     "止損後創新低再進場（新假設）": reentry}
for name, f in V.items():
    tr = pd.DataFrame([t | {"date": d["date"]} for d in nq for t in [f(d)] if t])
    x = tr[tr.date <= "2026-09-30"]; y = x.groupby(x.date.str[:4]).pnl.mean(); o = tr[tr.date >= "2026-10-01"]; t30 = tr[tr.date == "2026-10-30"]
    c = x.pnl.cumsum().values; dd = (np.maximum.accumulate(np.r_[0, c])[1:] - c).max() if len(x) else 0
    s30 = (f"{'空' if t30.side.iloc[0] == -1 else '多'} {hh(t30.ei.iloc[0])} SL {t30.risk.iloc[0]:.0f} 點 → {t30.res.iloc[0]} {hh(min(t30.xi.iloc[0], 389))} {t30.pnl.iloc[0]:+,.0f}") if len(t30) else "沒觸發"
    print(f"{name}\n   10/30：{s30}\n   歷史：{len(x)} 筆 勝率 {(x.pnl > 0).mean():.0%} 每筆 {x.pnl.mean():+.0f} 回撤 {dd:,.0f}｜2024/2025/2026 每筆 {'/'.join(f'{v:+.0f}' for v in y)}｜10 月 {len(o)} 筆 {o.pnl.sum():+,.0f}")
