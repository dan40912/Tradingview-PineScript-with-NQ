"""夜盤「先大跌、再拉回」的日子（像 2026-11-04 凌晨），美股盤中後來怎麼走、我們的規則表現如何。
條件在美東 01:00 判斷：夜盤（前一天 18:00 起）最低點比前一天 RTH 收盤低 ≥ X%，而且 01:00 的價格已經拉回跌幅的一半以上。"""
import pickle, numpy as np, pandas as pd, grid, strat
from eng import REPO
df = pd.read_csv(f"{REPO}/data/NQ_1m_continuous_20240102-20261019.csv.gz", usecols=["timestamp_utc", "high", "low", "close", "trade_date", "contract"])
et = pd.to_datetime(df.timestamp_utc).dt.tz_convert("America/New_York"); df["m"] = et.dt.hour * 60 + et.dt.minute
nq = pickle.load(open("NQ_days2.pkl", "rb")); D = {d["date"]: d for d in nq}
rth_close = df[(df.m >= 570) & (df.m <= 959)].groupby("trade_date").close.last()
rows = []
for td, g in df.groupby("trade_date"):
    if td not in D: continue
    prev = rth_close[rth_close.index < td]
    if not len(prev): continue
    pc = prev.iloc[-1]; on = g[(g.m >= 1080) | (g.m <= 60)]           # 前一天 18:00 到今天 01:00
    if len(on) < 100: continue
    lo = on.low.min(); px = on.close.iloc[-1]; drop = (pc - lo) / pc; rec = (px - lo) / (pc - lo) if pc > lo else 0
    d = D[td]; O, C = d["O"], d["C"]
    if np.isnan(O[0]): continue
    last = C[~np.isnan(C)][-1]
    t_i2 = strat.i2(d, last=89, tp=55, sl=145, xi=379, es=grid.ES); t_tr = strat.trend(d, ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="outpd", be=1.0)
    pos = "VAH 之上" if O[0] > d["vah"] else "VAL 之下" if O[0] < d["val"] else "區內"
    rows.append(dict(date=td, 夜盤跌幅=drop, 一點拉回=rec, 一點到開盤=O[0] - px, 開盤比前收=(O[0] / pc - 1), 開盤到收盤=last - O[0], 開盤位置=pos,
                     I2=t_i2["pnl"] if t_i2 else np.nan, 順勢=t_tr["pnl"] if t_tr else np.nan))
X = pd.DataFrame(rows)
for th in (0.01, 0.015):
    v = X[(X.夜盤跌幅 >= th) & (X.一點拉回 >= 0.5)]
    print(f"\n=== 夜盤跌 ≥ {th:.1%}、01:00 已拉回一半以上：{len(v)} 天（全部 {len(X)} 天）===")
    print(v.assign(夜盤跌幅=v.夜盤跌幅.map("{:.1%}".format), 一點拉回=v.一點拉回.map("{:.0%}".format), 開盤比前收=v.開盤比前收.map("{:+.1%}".format)).round(0).to_string(index=False))
    print(f"  01:00 到 09:30 開盤：繼續漲 {(v.一點到開盤 > 0).mean():.0%}，中位數 {v.一點到開盤.median():+.0f} 點")
    print(f"  美股盤中開盤到收盤：收紅 {(v.開盤到收盤 > 0).mean():.0%}，中位數 {v.開盤到收盤.median():+.0f} 點（所有日子收紅 {(X.開盤到收盤 > 0).mean():.0%}）")
    print(f"  開盤位置：{v.開盤位置.value_counts().to_dict()}｜I2（ES 同向）{v.I2.notna().sum()} 筆 合計 {v.I2.sum():+,.0f}｜順勢 {v.順勢.notna().sum()} 筆 合計 {v.順勢.sum():+,.0f}")
