"""逐分鐘照 Pine v14 寫法重跑「亞洲多 ES」，和 f_family.py 的「亞洲多＋ES同步」逐筆比對。"""
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib；放到該分支 research/ 下執行（見 docs/21 第 9 節）。
import pandas as pd, numpy as np
def load(sym):
    df = pd.read_csv(f'../data/{sym}_1m_continuous_20240102-20261110.csv.gz', usecols=['timestamp_utc','open','high','low','close','trade_date'])
    t = pd.to_datetime(df.timestamp_utc, utc=True).dt.tz_convert('America/New_York')
    df['m'] = t.dt.hour*60 + t.dt.minute; df['d'] = t.dt.strftime('%Y%m%d').astype(int); df['ts'] = t.astype('int64')//10**6
    return df
es = load('ES'); nq = load('NQ')
# ES stream -> esSig[evening day] = ts of first ES Asia-long reclaim (5m close)
esSig = {}; inR=False; sesLo=np.nan; esPdl=np.nan; asLo=np.nan; day=None
for m,d,ts,h,l,c in zip(es.m.values, es.d.values, es.ts.values, es.high.values, es.low.values, es.close.values):
    rth = 570<=m<=959
    if inR and not rth: esPdl=sesLo; inR=False
    if rth:
        if not inR: sesLo=l; inR=True
        else: sesLo=min(sesLo,l)
    if m>=1080 and day!=d: day=d; asLo=l
    elif (m>=1080 or m<=179) and not np.isnan(asLo): asLo=min(asLo,l)
    if m%5==4 and (m>=1084 or m<=119) and day not in esSig and not np.isnan(esPdl) and not np.isnan(asLo):
        if asLo<esPdl and c>esPdl: esSig[day]=ts
# NQ stream, long side only
out=[]; inR=False; sesLo=np.nan; pdl=np.nan; asLo=np.nan; day=None; st=0; sig=-1; E=np.nan; k=None
O,H,L,C,M,D,TS,TD = nq.open.values,nq.high.values,nq.low.values,nq.close.values,nq.m.values,nq.d.values,nq.ts.values,nq.trade_date.values
for j in range(len(nq)):
    m=M[j]; rth=570<=m<=959
    if inR and not rth: pdl=sesLo; inR=False
    if rth:
        if not inR: sesLo=L[j]; inR=True
        else: sesLo=min(sesLo,L[j])
    if m>=1080 and day!=D[j]: day=D[j]; asLo=L[j]; st=0
    elif (m>=1080 or m<=179) and not np.isnan(asLo): asLo=min(asLo,L[j])
    if np.isnan(asLo): continue
    if st==1 and j>sig:
        E=O[j]; st=2; out.append(dict(date=TD[j],entry=E,res=None)); k=len(out)-1
    if st==2:
        res=None
        if L[j]<=E-145: res=-2948
        elif H[j]>=E+55: res=1052
        elif 179<=m<1080: res=(C[j]-E)*20-48
        if res is not None: out[k]['res']=res; st=3
    if m%5==4 and (m>=1084 or m<=119) and st==0 and not np.isnan(pdl) and asLo<pdl and C[j]>pdl:
        ok = day in esSig and esSig[day]<=TS[j]
        st = 1 if ok else 3; sig=j
P=pd.DataFrame(out)
F=pd.read_csv('f_family_trades.csv'); F=F[F.rule=='亞洲多＋ES同步'][['date','entry','pnl']]
mm=F.merge(P,on='date',how='outer',indicator=True,suffixes=('_bt','_pine'))
print(mm._merge.value_counts().to_dict())
b=mm[mm._merge=='both']; print('entry diff',(abs(b.entry_bt-b.entry_pine)>0.01).sum(),'pnl diff',(abs(b.pnl-b.res)>1).sum(),'of',len(b))
print(mm[mm._merge!='both'].to_string())
