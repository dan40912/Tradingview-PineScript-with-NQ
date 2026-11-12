import pandas as pd, numpy as np
F = pd.read_csv('/home/user/Tradingview-PineScript-with-NQ/research/winrate_recent/wr_table.csv.gz')
F["key"] = F.rule + "|" + F.side
lowv = F.vol < 1.2
HW = (F.key == "I2|多") | ((F.key == "ASIA-PD-k1|多") & (F.es_ok == True)) | ((F.key == "ASIA-VA-k1|多") & (F.es_ok == True)) | ((F.key == "AM-ON-k1|多") & (F.es_ok == True) & ~lowv)
T = F[HW].sort_values(["date", "entry_slot"]).copy()
a = T.key.str.startswith("ASIA"); T = pd.concat([T[~a], T[a].drop_duplicates("date")]).sort_values(["date", "entry_slot"])
keep = []
for d, x in T.groupby("date", sort=False):
    closed = []
    for t in x.itertuples():
        if sum(p for xs, p in closed if xs <= t.entry_slot) <= -2900: keep.append(False); continue
        keep.append(True); closed.append((t.exit_slot, t.pnl55))
T = T[keep]
# 每天美股盤中漲跌（收盤對前一天收盤）
fs = ['/home/user/Tradingview-PineScript-with-NQ/data/NQ_1m_continuous_20240102-20261019.csv.gz', '/home/user/Tradingview-PineScript-with-NQ/data/NQ_1m_yahoo_20261020-20261110.csv.gz']
D = pd.concat([pd.read_csv(f, usecols=['trade_date', 'close', 'is_rth']) for f in fs])
c = D[D.is_rth == 1].groupby('trade_date').close.last(); ret = (c / c.shift(1) - 1) * 100
T["ret"] = T.date.map(ret); T["prev_ret"] = T.date.map(ret.shift(1))
print("=== 11/04～11/09 高勝率組合的交易（trade_date＝美東 18:00 起的交易日）")
print(T[(T.date >= "2026-11-04") & (T.date <= "2026-11-10")][["date", "key", "entry_slot", "why55", "pnl55"]].to_string(index=False))
P = F[(F.date >= "2026-11-04") & (F.date <= "2026-11-07") & F.key.isin(["ASIA-PD-k1|多", "ASIA-VA-k1|多", "AM-ON-k1|多", "I2|多"])]
print("\n=== 同期間沒過濾的同類做多點（看 ES／波動擋掉了什麼）")
print(P[["date", "key", "es_ok", "vol", "why55", "pnl55"]].round(2).to_string(index=False))
print("\n=== 歷史：當天美股跌幅分組")
T["bucket"] = pd.cut(T.ret, [-99, -2.5, -1.5, -0.5, 0.5, 99], labels=["跌>2.5%", "跌1.5~2.5%", "跌0.5~1.5%", "平盤", "漲"])
print(T.groupby("bucket", observed=False).pnl55.agg(['size', lambda p: (p > 0).mean() * 100, 'mean', 'sum']).round(0).to_string())
T["pb"] = pd.cut(T.prev_ret, [-99, -2.5, -1.5, 99], labels=["前一天跌>2.5%", "前一天跌1.5~2.5%", "其他"])
print(T.groupby("pb", observed=False).pnl55.agg(['size', lambda p: (p > 0).mean() * 100, 'mean', 'sum']).round(0).to_string())
dy = T.groupby("date").pnl55.sum(); wk = T.groupby(pd.to_datetime(T.date).dt.to_period("W")).pnl55.sum(); mo = T.groupby(T.date.str[:7]).pnl55.sum()
s = (T.pnl55 < 0).astype(int); runs = s.groupby((s != s.shift()).cumsum()).sum().max()
print(f"\n最差一天 {dy.min():+,.0f}（{dy.idxmin()}）｜最差一週 {wk.min():+,.0f}（{wk.idxmin()}）｜最差一個月 {mo.min():+,.0f}（{mo.idxmin()}）｜最多連續虧損 {runs} 筆｜虧損月數 {(mo<0).sum()}/{len(mo)}")
print("最差 5 天：", dy.nsmallest(5).round(0).to_dict())
big = ret[ret <= -2.5].index; print("\n跌 >2.5% 的日子共", len(big), "天；高勝率組合在這些日子有交易的：", sorted(set(T.date) & set(big)))

# 同時最多一筆持倉（前一筆還沒出場就不進新單）
T2 = F[HW].sort_values(["date", "entry_slot"]).copy()
a = T2.key.str.startswith("ASIA"); T2 = pd.concat([T2[~a], T2[a].drop_duplicates("date")]).sort_values(["date", "entry_slot"])
keep = []
for d, x in T2.groupby("date", sort=False):
    busy = -1; real = 0
    for t in x.itertuples():
        if t.entry_slot <= busy or real <= -2900: keep.append(False); continue
        keep.append(True); busy = t.exit_slot; real += t.pnl55
T2 = T2[keep]
def dd(p): c = p.cumsum(); return (c - c.cummax()).min()
for name, X in [("原本", T), ("同時最多一筆", T2)]:
    dy = X.groupby("date").pnl55.sum(); X = X.copy(); X["Y"] = np.select([X.date >= "2026-10-01", X.date >= "2026-01-01", X.date >= "2025-01-01"], ["近期", "2026-1~9", "2025"], "2024")
    print(f"{name}：{len(X)} 筆｜勝率 {(X.pnl55>0).mean()*100:.1f}%｜合計 {X.pnl55.sum():+,.0f}｜最大回撤 {dd(X.pnl55):+,.0f}｜最差一天 {dy.min():+,.0f}｜逐年合計 {X.groupby('Y').pnl55.sum().round(0).to_dict()}")

# 下單前檢查：已實現損益＋這筆最壞（−$2,948）＋同時持有的單最壞 < −$3,000 就不下單
T3 = F[HW].sort_values(["date", "entry_slot"]).copy()
a = T3.key.str.startswith("ASIA"); T3 = pd.concat([T3[~a], T3[a].drop_duplicates("date")]).sort_values(["date", "entry_slot"])
keep = []
for d, x in T3.groupby("date", sort=False):
    closed = []; opened = []
    for t in x.itertuples():
        real = sum(p for xs, p in closed if xs <= t.entry_slot)
        open_n = sum(1 for xs in opened if xs > t.entry_slot)
        if real - 2948 * (open_n + 1) < -3000: keep.append(False); continue
        keep.append(True); closed.append((t.exit_slot, t.pnl55)); opened.append(t.exit_slot)
T3 = T3[keep]
X = T3.copy(); X["Y"] = np.select([X.date >= "2026-10-01", X.date >= "2026-01-01", X.date >= "2025-01-01"], ["近期", "2026-1~9", "2025"], "2024")
dy = X.groupby("date").pnl55.sum(); wk = X.groupby(pd.to_datetime(X.date).dt.to_period("W")).pnl55.sum()
print(f"下單前算最壞：{len(X)} 筆｜勝率 {(X.pnl55>0).mean()*100:.1f}%｜合計 {X.pnl55.sum():+,.0f}｜最大回撤 {dd(X.pnl55):+,.0f}｜最差一天 {dy.min():+,.0f}｜最差一週 {wk.min():+,.0f}｜逐年 {X.groupby('Y').pnl55.sum().round(0).to_dict()}｜逐年勝率 {X.groupby('Y').pnl55.apply(lambda p:(p>0).mean()*100).round(1).to_dict()}")
