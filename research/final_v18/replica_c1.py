"""照 Pine v18 寫法重跑 C1 開盤 5 分鐘驅動與第二層 M（有 C1 做 C1，否則 TD10），和 A_trend、C_indicators 逐筆比對。"""
import numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad'
df = pd.read_csv(f'{SP}/wt/data/NQ_1m_continuous_20240102-20261110.csv.gz', usecols=['timestamp_utc','open','high','low','close','trade_date','is_rth'])
t = pd.to_datetime(df.timestamp_utc, utc=True).dt.tz_convert('America/New_York'); df['m'] = t.dt.hour*60 + t.dt.minute
TD10 = pd.read_csv('replica_td10_daily.csv').set_index('date').td.to_dict()
rngs = []; out = []
for d, x in df.groupby('trade_date', sort=True):
    r = x[(x.m >= 570) & (x.m <= 959)]
    if len(r) == 0: continue
    adr = np.mean(rngs[-10:]) if len(rngs) >= 10 else np.nan
    O = dict(zip(r.m, r.open)); H = dict(zip(r.m, r.high)); L = dict(zip(r.m, r.low)); C = dict(zip(r.m, r.close))
    mins = sorted(O)
    def trade(kind, e_min, d_, stop_dist, trail):
        e = next((k for k in mins if k >= e_min), None)
        if e is None or e > 955: return None
        px = O[e]; stop = px - d_ * stop_dist; best = 0.0; trailed = False
        for k in [k for k in mins if k >= e]:
            h, l = H[k], L[k]
            if (d_ == 1 and l <= stop) or (d_ == -1 and h >= stop):
                xp = stop if (k == e or (d_ == 1 and O[k] > stop) or (d_ == -1 and O[k] < stop)) else O[k]
                xp = min(xp, stop) if d_ == 1 else max(xp, stop)
                return dict(date=d, kind=kind, dir=d_, entry=px, exit=xp, why='移動停損' if trailed else '止損', x=k)
            if k >= 955:
                return dict(date=d, kind=kind, dir=d_, entry=px, exit=C[k], why='15:55', x=k)
            fav = (h - px) if d_ == 1 else (px - l); best = max(best, fav)
            if trail and best >= 100:
                ns = px + d_ * (best - 100)
                if (d_ == 1 and ns > stop) or (d_ == -1 and ns < stop): stop = ns; trailed = True
        k = mins[-1]; return dict(date=d, kind=kind, dir=d_, entry=px, exit=C[k], why='資料結束', x=k)
    c1 = None
    if adr == adr and 574 in C and 570 in O:
        mv = C[574] - O[570]
        if abs(mv) >= 0.15 * adr and mv != 0:
            c1 = trade('C1', 575, 1 if mv > 0 else -1, min(max(0.25 * adr, 10.0), 145.0), True)
            if c1: c1['why_entry'] = f'開盤 5 分鐘 {mv:+.2f} 點（門檻 {0.15*adr:.2f}）'
    if c1: out.append(c1)
    elif TD10.get(d, 0) != 0:
        tdt = trade('TD10', 600, int(TD10[d]), 100.0, False)
        if tdt: out.append(tdt)
    if r.m.max() > 779:   # 美股盤中有交易到 12:59 之後才算正常交易日（排除假日半天）
        rngs.append(r.high.max() - r.low.min())
T = pd.DataFrame(out); T['pts'] = T.dir * (T.exit - T.entry); T['pnl_nq'] = T.pts * 20 - 48
T.to_csv('replica_tier2_trades.csv', index=False)
A = pd.read_csv(f'{SP}/agents/A_trend/trades_C1_開盤驅動5分.csv')
c = T[T.kind == 'C1'].merge(A[['date', 'd', 'entry', 'exit', 'pnl']], on='date', how='outer', indicator=True)
both = c[c._merge == 'both']
print('C1：A 的', len(A), '筆｜Pine 寫法', int((T.kind == 'C1').sum()), '筆｜兩邊都有', len(both),
      '｜進場價不同', int((abs(both.entry_x - both.entry_y) > 0.01).sum()), '｜損益差 >$1', int((abs(both.pnl_nq - both.pnl) > 1).sum()))
print(' 只有 Pine：', c[c._merge == 'left_only'].date.tolist()[:12]); print(' 只有 A：', c[c._merge == 'right_only'].date.tolist()[:12])
bad = both[abs(both.pnl_nq - both.pnl) > 1]; print(bad[['date', 'entry_x', 'exit_x', 'exit_y', 'pnl_nq', 'pnl']].head(10).to_string())
