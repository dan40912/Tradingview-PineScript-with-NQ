"""多市場 scanner：同時看 NQ 和 ES，誰先出現組合 D 的訊號就做誰（同時只持 1 筆）。
ES 的價格 ×3.7 換成 NQ 等值點數，所以損益是「1 口 NQ 等值」；I2 的同向確認在 ES 上改用 NQ。"""
import pickle, numpy as np, pandas as pd, grid, strat
nq = pickle.load(open("NQ_days2.pkl", "rb")); es = pickle.load(open("ES_days2.pkl", "rb"))
NQd = {d["date"]: (-1 if d["O"][0] > d["vah"] else 1 if d["O"][0] < d["val"] else 0) for d in nq if not np.isnan(d["O"][0])}
T1 = dict(ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="outpd", be=1.0)
def sig(days, conf, mk):
    out = {}
    for d in days:
        for t in (strat.i2(d, last=89, tp=55, sl=145, xi=379, es=conf), strat.trend(d, **T1)):
            if t: out.setdefault(d["date"], []).append(t | {"date": d["date"], "mkt": mk})
    return out
S = {"NQ": sig(nq, grid.ES, "NQ"), "ES": sig(es, NQd, "ES")}
def port(mkts):
    out = []
    for d in sorted(set().union(*[S[m].keys() for m in mkts])):
        busy, n, pl, losses = -1, 0, 0.0, 0
        for t in sorted([t for m in mkts for t in S[m].get(d, [])], key=lambda t: (t["ei"], t["mkt"] != "NQ")):
            if t["ei"] <= busy or n >= 3 or pl <= -3000 or losses >= 2: continue
            out.append(t); busy = t["xi"]; n += 1; pl += t["pnl"]; losses += t["pnl"] < 0
    return pd.DataFrame(out)
for mk in (["NQ"], ["ES"], ["NQ", "ES"]):
    df = port(mk); x = df[df.date <= "2026-09-30"]; c = x.pnl.cumsum().values; dd = (np.maximum.accumulate(np.r_[0, c])[1:] - c).max()
    y = x.groupby(x.date.str[:4]).pnl.mean(); mo = x.groupby(x.date.str[:7]).pnl.sum(); o = df[df.date >= "2026-10"]
    print(f"{'＋'.join(mk):<6} {len(x):3d} 筆 每筆 {x.pnl.mean():+4.0f} 合計 {x.pnl.sum():+7,.0f} 回撤 {dd:6,.0f} 比 {x.pnl.sum()/dd:4.1f} 虧錢月 {(mo<0).sum():2d}｜"
          f"{y.get('2024',0):+4.0f}/{y.get('2025',0):+4.0f}/{y.get('2026',0):+4.0f}｜10月 {len(o)} 筆 {o.pnl.sum():+,.0f}"
          + (f"｜其中 ES {int((x.mkt=='ES').sum())} 筆" if len(mk) > 1 else ""))
