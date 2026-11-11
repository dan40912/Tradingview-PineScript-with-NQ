"""逐分鐘照 Pine v13 亞洲盤參考點的寫法重跑，和 family.py 的 ASIA-PD-k1 逐筆比對。"""
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib；放到該分支 research/ 下執行（見 docs/21 第 10 節）。
import pandas as pd, numpy as np
df = pd.read_csv('../data/NQ_1m_continuous_20240102-20261110.csv.gz', usecols=['timestamp_utc','open','high','low','close','trade_date'])
t = pd.to_datetime(df.timestamp_utc, utc=True).dt.tz_convert('America/New_York')
df['m'] = t.dt.hour*60 + t.dt.minute; df['d'] = t.dt.strftime('%Y-%m-%d')
O,H,L,C,M,D,TD = df.open.values, df.high.values, df.low.values, df.close.values, df.m.values, df.d.values, df.trade_date.values
pdh=pdl=np.nan; sesHi=sesLo=np.nan; inR=False; asHi=asLo=np.nan; asDay=None
st=[0,0]; sig=[0,0]; E=[np.nan,np.nan]; out=[]
for j in range(len(df)):
    m=M[j]; rth = 570<=m<=959
    if inR and not rth:          # RTH ended -> pdh/pdl
        pdh,pdl=sesHi,sesLo; inR=False
    if rth:
        if not inR: sesHi,sesLo=H[j],L[j]; inR=True
        else: sesHi=max(sesHi,H[j]); sesLo=min(sesLo,L[j])
    inAsia = m>=1080 or m<=179
    if m>=1080 and asDay!=D[j]:
        asDay=D[j]; asHi,asLo=H[j],L[j]; st=[0,0]
    elif inAsia and not np.isnan(asHi):
        asHi=max(asHi,H[j]); asLo=min(asLo,L[j])
    if np.isnan(asHi): continue
    for r in (0,1):
        sd=-1 if r==0 else 1
        if st[r]==1 and j>sig[r]:
            E[r]=O[j]; st[r]=2; ent=(TD[j],sd,O[j])
            out.append(dict(date=TD[j],dir=sd,entry=O[j],res=None)); k=len(out)-1; st_idx=k
            if r==0: k0=k
            else: k1=k
        if st[r]==2:
            e=E[r]; k = k0 if r==0 else k1; res=None
            if (L[j]<=e-145 if sd==1 else H[j]>=e+145): res=-145*20-48
            elif (H[j]>=e+55 if sd==1 else L[j]<=e-55): res=55*20-48
            elif 179<=m<1080: res=sd*(C[j]-e)*20-48
            if res is not None: out[k]['res']=res; st[r]=3
    if m%5==4 and (m>=1084 or m<=119):
        if not np.isnan(pdh) and st[0]==0 and asHi>pdh and C[j]<pdh: st[0]=1; sig[0]=j
        if not np.isnan(pdl) and st[1]==0 and asLo<pdl and C[j]>pdl: st[1]=1; sig[1]=j
P=pd.DataFrame(out)
F=pd.read_csv('family_trades_NQ.csv'); F=F[F.rule=='ASIA-PD-k1'][['date','dir','entry','pnl']]
m=F.merge(P,on=['date','dir'],how='outer',indicator=True,suffixes=('_bt','_pine'))
print(m._merge.value_counts().to_dict())
b=m[m._merge=='both']; print('entry diff',(abs(b.entry_bt-b.entry_pine)>0.01).sum(),'pnl diff',(abs(b.pnl-b.res)>1).sum(),'of',len(b))
print(m[m._merge!='both'].head(20).to_string())
print(b[(abs(b.pnl-b.res)>1)|(abs(b.entry_bt-b.entry_pine)>0.01)].head(10).to_string())
