"""逐分鐘照 Pine v18 的寫法重跑第二層（C1 開盤動能、TD10 趨勢日）：
5 分K 由 1 分K 組成（時鐘對齊）、EMA／ATR／DI／MACD 逐根遞迴；和 A_trend、C_indicators 的結果逐筆／逐日比對。"""
import numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad'
df = pd.read_csv(f'{SP}/wt/data/NQ_1m_continuous_20240102-20261110.csv.gz', usecols=['timestamp_utc','open','high','low','close','volume','contract','trade_date'])
t = pd.to_datetime(df.timestamp_utc, utc=True).dt.tz_convert('America/New_York')
df['m'] = t.dt.hour*60+t.dt.minute; df['d'] = t.dt.strftime('%Y%m%d').astype(int)
# 換月調整（和研究一樣，讓連續序列沒有跳空；Pine 用單一合約本來就沒有）
days = df.trade_date.unique(); con = df.groupby('trade_date').contract.first()
first_o = df.groupby('trade_date').open.first(); last_c = df.groupby('trade_date').close.last()
off = np.zeros(len(days))
for k in range(1, len(days)):
    if con[days[k]] != con[days[k-1]]: off[k] = first_o[days[k]] - last_c[days[k-1]]
cum = np.concatenate([np.cumsum(off[::-1])[::-1][1:], [0.0]]); adj = dict(zip(days, cum))
A = df.trade_date.map(adj).values
O,H,L,C,V,M,TD = df.open.values,df.high.values,df.low.values,df.close.values,df.volume.values,df.m.values,df.trade_date.values
Ha, La, Ca = H+A, L+A, C+A
TH = dict(effi=0.3215553652062646, above_vw=0.8, ema_slope=1.3001621620137531, di=21.74334553752221, macdh=0.3815786106438511)
# 5 分K 遞迴狀態
a20, a12, a26, a9, aw = 2/21, 2/13, 2/27, 2/10, 1/14
st = dict(e20=None, e12=None, e26=None, sig=None, atr=None, pdm=None, ndm=None, pc=None, ph=None, pl=None)
e20hist = []
cur = None   # (date, bucket, h, l, c)
feat_at = {}  # trade_date -> dict at 10:00
def close_bucket(b):
    h, l, c = b[2], b[3], b[4]
    s = st
    if s['pc'] is None:
        tr = h - l; upm = dnm = np.nan
    else:
        tr = max(h - l, abs(h - s['pc']), abs(l - s['pc'])); upm = h - s['ph']; dnm = s['pl'] - l
    pdm = upm if (upm == upm and upm > dnm and upm > 0) else 0.0
    ndm = dnm if (dnm == dnm and dnm > upm and dnm > 0) else 0.0
    s['atr'] = tr if s['atr'] is None else s['atr'] + aw*(tr - s['atr'])
    s['pdm'] = pdm if s['pdm'] is None else s['pdm'] + aw*(pdm - s['pdm'])
    s['ndm'] = ndm if s['ndm'] is None else s['ndm'] + aw*(ndm - s['ndm'])
    for k, a in (('e20', a20), ('e12', a12), ('e26', a26)):
        s[k] = c if s[k] is None else s[k] + a*(c - s[k])
    macd = s['e12'] - s['e26']
    s['sig'] = macd if s['sig'] is None else s['sig'] + a9*(macd - s['sig'])
    e20hist.append(s['e20']); del e20hist[:-7]
    s['pc'], s['ph'], s['pl'] = c, h, l
    s['last'] = dict(ema_slope=(s['e20'] - e20hist[0]) / s['atr'] if len(e20hist) == 7 else np.nan,
                     di=100*s['pdm']/s['atr'] - 100*s['ndm']/s['atr'], macdh=(macd - s['sig']) / s['atr'])
# 開盤 30 分鐘的效率、VWAP 時間
o930 = None; path = 0.0; prevc = None; pv = vv = 0.0; above = []; 
for j in range(len(df)):
    m = M[j]; bk = (TD[j], m // 5 if m < 1080 else (m // 5))
    key = (TD[j], m // 5)
    if cur is not None and (cur[0], cur[1]) != key:
        close_bucket(cur); cur = None
    if m == 600:   # 10:00：先關掉 09:55 那根，再判斷
        lastf = st.get('last', {})
        if o930 is not None:
            f = dict(effi=(prevc - o930) / path if path > 0 else 0.0, above_vw=np.mean(above) * 2 - 1, **lastf)
            up = sum(1 for k, v in f.items() if v == v and v > TH[k]); dn = sum(1 for k, v in f.items() if v == v and v < -TH[k])
            feat_at[TD[j]] = dict(f, up=up, dn=dn, td=1 if up >= 3 and dn == 0 else -1 if dn >= 3 and up == 0 else 0)
    if m == 570:
        o930 = O[j]; path = 0.0; prevc = o930; pv = vv = 0.0; above = []
    if 570 <= m < 600 and o930 is not None:
        path += abs(C[j] - prevc); prevc = C[j]
        pv += (H[j] + L[j] + C[j]) / 3 * V[j]; vv += V[j]
        above.append(1.0 if C[j] > pv / vv else 0.0)
    if cur is None: cur = [key[0], key[1], Ha[j], La[j], Ca[j]]
    else: cur[2] = max(cur[2], Ha[j]); cur[3] = min(cur[3], La[j]); cur[4] = Ca[j]
R = pd.DataFrame.from_dict(feat_at, orient='index'); R.index.name = 'date'; R.to_csv('replica_td10_daily.csv')
ref = pd.read_csv(f'{SP}/agents/C_indicators/out_td_daily.csv').set_index('date')
j = ref.join(R, how='inner'); j = j[j.index >= '2024-02-01']
print('TD10（q80 k3）逐日比對：', int((j.TD10_q80k3 == j.td).sum()), '/', len(j), '｜研究有訊號', int((j.TD10_q80k3 != 0).sum()), '天，Pine 寫法有', int((j.td != 0).sum()), '天')
print(j[j.TD10_q80k3 != j.td][['TD10_q80k3', 'td', 'up', 'dn', 'effi', 'above_vw', 'ema_slope', 'di', 'macdh']].round(3).head(15).to_string())
