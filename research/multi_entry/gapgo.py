import pandas as pd, numpy as np, glob
fs=['/home/user/Tradingview-PineScript-with-NQ/data/NQ_1m_continuous_20240102-20261019.csv.gz','/home/user/Tradingview-PineScript-with-NQ/data/NQ_1m_yahoo_20261020-20261110.csv.gz']
df=pd.concat([pd.read_csv(f,usecols=['timestamp_utc','open','high','low','close','contract','trade_date','is_rth']) for f in fs],ignore_index=True)
t=pd.to_datetime(df.timestamp_utc,utc=True).dt.tz_convert('America/New_York'); df['m']=t.dt.hour*60+t.dt.minute
df['sm']=np.where(df.m>=1080,df.m-1080,df.m+360)
days=sorted(df.trade_date.unique()); G={d:x.sort_values('sm').reset_index(drop=True) for d,x in df.groupby('trade_date')}
def sim(x,i0,d,tend,tp=55,sl=145):
    e=x.open[i0]
    for i in range(i0,len(x)):
        if x.sm[i]>tend: return (x.close[i-1]-e)*d
        if (x.low[i]<=e-sl if d==1 else x.high[i]>=e+sl): return -sl
        if (x.high[i]>=e+tp if d==1 else x.low[i]<=e-tp): return tp
    return (x.close.iloc[-1]-e)*d
rows=[]
for k in range(1,len(days)):
    a,b=G[days[k-1]],G[days[k]]
    ra=a[a.is_rth==1]; 
    if len(ra)<300 or a.contract.iloc[-1]!=b.contract.iloc[0]: continue
    pc=ra.close.iloc[-1]
    pre=b[(b.sm<930)]; op=b[b.sm>=930]
    if len(pre)<500 or len(op)<200: continue
    chg=pre.close.iloc[-1]-pc; pct=chg/pc*100
    i0=op.index[0]
    r=dict(date=days[k],chg=chg,pct=pct)
    r['open_long']=sim(b,i0,1,1109); r['open_short']=sim(b,i0,-1,1109)
    # 隔夜追多：夜盤第一次漲到前收 +200 點（美東 08:00 前）就做多，08:29 前出場
    hit=pre[(pre.high>=pc+200)&(pre.sm<=840)]
    if len(hit): r['chase200']=sim(b,hit.index[0]+1,1,869)
    rows.append(r)
R=pd.DataFrame(rows); R['yr']=R.date.str[:4]
for c in ['open_long','open_short','chase200']: R[c]=R[c]*20-48
pd.set_option('display.width',200)
for th in [200,300,400,500]:
    x=R[R.chg>=th]; y=R[R.chg<=-th]
    print(f'夜盤漲 ≥{th} 點：{len(x)} 天｜開盤做多每筆 {x.open_long.mean():.0f}（勝率 {(x.open_long>0).mean()*100:.0f}%）｜開盤做空每筆 {x.open_short.mean():.0f}｜逐年做多', x.groupby('yr').open_long.agg(['size','mean']).round(0).values.tolist())
    print(f'   夜盤跌 ≥{th} 點：{len(y)} 天｜開盤做空每筆 {y.open_short.mean():.0f}（勝率 {(y.open_short>0).mean()*100:.0f}%）')
c=R.dropna(subset=['chase200']); print('夜盤漲到前收+200 就追多：',len(c),'筆｜每筆',round(c.chase200.mean()),'｜勝率',round((c.chase200>0).mean()*100),'%｜逐年',c.groupby('yr').chase200.agg(['size','mean']).round(0).values.tolist())
x=R[R.pct>=1.5]; print('夜盤漲 ≥1.5%：',len(x),'天｜開盤做多每筆',round(x.open_long.mean()),'勝率',round((x.open_long>0).mean()*100),'%')
R.to_csv('gapgo.csv',index=False)
