"""黑天鵝：停損被跳過多少點、最大 1／5 分鐘波動、週末跳空、極端日。研究用途，不是投資建議。"""
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib，以及 research/winrate_recent/wr_table.csv.gz；放到該分支 research/ 下執行（見 docs/23）。
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from lib import data
g = data.load_grid("NQ", include_holdout=True)
F = pd.read_csv("wr_table.csv.gz")
# 1. 止損那根 1 分K 的開盤已經超過止損價（跳過去）→ 實際成交會更差
rows = []
for t in F[F.why55 == "sl"].itertuples():
    i = g.dindex[t.date]; O, H, L = g.O[i], g.H[i], g.L[i]; e = t.entry_slot
    while np.isnan(O[e]): e += 1
    px = O[e]; sl = px - t.dir * 145
    for s in range(e, t.exit_slot + 1):
        if np.isnan(H[s]): continue
        if (t.dir == 1 and L[s] <= sl) or (t.dir == -1 and H[s] >= sl):
            gap = max(0.0, (sl - O[s]) if t.dir == 1 else (O[s] - sl)) if s > e else 0.0
            rows.append(dict(date=t.date, rule=t.rule, gap=gap)); break
G = pd.DataFrame(rows)
print(f"1. 止損 {len(G)} 次｜開盤就跳過止損價的 {int((G.gap>0).sum())} 次（{(G.gap>0).mean()*100:.1f}%）｜跳過幅度：平均 {G[G.gap>0].gap.mean():.1f} 點、95% {G.gap.quantile(.95):.1f}、99% {G.gap.quantile(.99):.1f}、最大 {G.gap.max():.1f} 點（{G.loc[G.gap.idxmax(),'date']}）")
# 2. 最大 1 分、5 分鐘波動
C = g.C.ravel(); H = g.H.ravel(); L = g.L.ravel()
r1 = np.nanmax(g.H - g.L, axis=1)
one = pd.Series(np.nanmax(g.H - g.L, axis=1), index=g.dates)
print(f"2. 單一 1 分K 最大振幅：{one.max():.1f} 點（{one.idxmax()}）；前 5 名 {one.nlargest(5).round(1).to_dict()}")
c5 = pd.DataFrame(g.C).T.rolling(5).apply(lambda x: x[-1] - x[0], raw=True).T
m5 = pd.Series(np.nanmin(c5.values, axis=1), index=g.dates)
print(f"   5 分鐘內最大跌幅：{m5.min():.1f} 點（{m5.idxmin()}）；前 5 名 {m5.nsmallest(5).round(1).to_dict()}")
# 3. 週末／假日跳空（前一個交易日最後一根 → 下一個交易日第一根）
gaps = []
for k in range(1, len(g.dates)):
    a = g.C[k - 1][~np.isnan(g.C[k - 1])]; b = g.O[k][~np.isnan(g.O[k])]
    if len(a) and len(b) and g.contract[k] == g.contract[k - 1]: gaps.append((g.dates[k], b[0] - a[-1], (b[0] / a[-1] - 1) * 100))
GP = pd.DataFrame(gaps, columns=["date", "pts", "pct"])
print(f"3. 交易日之間的跳空：最大跌 {GP.pts.min():.1f} 點（{GP.pct.min():.2f}%，{GP.loc[GP.pts.idxmin(),'date']}）、最大漲 {GP.pts.max():.1f} 點；超過 145 點的 {int((GP.pts.abs()>145).sum())} 次")
# 4. 極端日：美股盤中跌幅
last = np.array([r[~np.isnan(r)][-1] if np.isfinite(r).any() else np.nan for r in g.rth(g.C)])
cl = pd.Series(last, index=g.dates)[g.has_rth]; ret = (cl / cl.shift(1) - 1) * 100
print(f"4. 美股盤中收盤最大單日跌幅：{ret.min():.2f}%（{ret.idxmin()}）；跌超過 4% 的日子 {ret[ret<=-4].round(2).to_dict()}")
G.to_csv("swan_stop_gaps.csv", index=False)
