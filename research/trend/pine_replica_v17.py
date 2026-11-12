"""逐分鐘照 Pine v17 的寫法重跑：亞洲 VAL 多 ES、早盤夜低多 ES、前高空／前低多的 VWAP 順勢判斷、暴衝幅度，和回測表逐筆比對。"""
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib，以及 research/winrate_recent/wr_table.csv.gz；放到該分支 research/ 下執行（見 docs/23）。
import pandas as pd, numpy as np
BIN = 5.0
def load(sym):
    df = pd.read_csv(f'../data/{sym}_1m_continuous_20240102-20261110.csv.gz', usecols=['timestamp_utc','open','high','low','close','volume','trade_date'])
    t = pd.to_datetime(df.timestamp_utc, utc=True).dt.tz_convert('America/New_York')
    df['m'] = t.dt.hour*60 + t.dt.minute; df['d'] = t.dt.strftime('%Y%m%d').astype(int); df['ts'] = t.astype('int64')//10**6
    return df
def va(prof):
    ks = sorted(prof); kmin, kmax = ks[0], ks[-1]; tot = sum(prof.values())
    poc = max(ks, key=lambda k: (prof[k], -k)); lo = hi = poc; acc = prof[poc]
    while acc < 0.7*tot and (lo > kmin or hi < kmax):
        lv = prof.get(lo-1, -1.0); hv = prof.get(hi+1, -1.0)
        if hv > lv: hi += 1; acc += max(0, hv)
        else: lo -= 1; acc += max(0, lv)
    return lo*BIN
def es_signals():
    es = load('ES'); out = {'va': {}, 'on': {}}
    inR=False; prof={}; VAL=np.nan; asLo=np.nan; onLo=np.nan; rthLo=np.nan; day=None
    for m,d,ts,h,l,c,v in zip(es.m.values, es.d.values, es.ts.values, es.high.values, es.low.values, es.close.values, es.volume.values):
        rth = 570<=m<=959
        if inR and not rth:
            inR=False
            if prof: VAL=va(prof)
            prof={}
        if rth:
            inR=True; a=int(np.floor(l/BIN)); b=int(np.floor(h/BIN))
            for k in range(a,b+1): prof[k]=prof.get(k,0)+v/(b-a+1)
            rthLo = l if (m==570 or np.isnan(rthLo)) else min(rthLo,l)
        if m>=1080 and day!=d: day=d; asLo=l; onLo=l; rthLo=np.nan
        else:
            if (m>=1080 or m<=179) and not np.isnan(asLo): asLo=min(asLo,l)
            if (m>=1080 or m<570) and not np.isnan(onLo): onLo=min(onLo,l)
        if m%5==4 and (m>=1084 or m<=119) and day not in out['va'] and not np.isnan(VAL) and not np.isnan(asLo):
            if asLo<VAL and c>VAL: out['va'][day]=ts
        ei=m+1-570
        if m%5==4 and 574<=m<=689 and not (28<=ei<=32) and day not in out['on'] and not np.isnan(onLo) and not np.isnan(rthLo):
            if rthLo<onLo and c>onLo: out['on'][day]=ts
    return out
E = es_signals()
nq = load('NQ')
O,H,L,C,V,M,D,TS,TD = [nq[k].values for k in ['open','high','low','close','volume','m','d','ts','trade_date']]
inR=False; prof={}; VAL=np.nan; onLo=np.nan; onDay=None; dayLo=np.nan; xDay=None
st=[0,0]; sig=[-1,-1]; xs=[]; pv=vv=0.0; recent=[]; vwprev_at={}; shock_at={}
for j in range(len(nq)):
    m=M[j]; rth=570<=m<=959; i1=m-570
    if m==570: pv=vv=0.0
    vwprev_at[j] = pv/vv if vv>0 else np.nan
    w=recent; shock_at[j] = max((abs(w[k+5]-w[k]) for k in range(len(w)-5)), default=0.0)
    if inR and not rth:
        inR=False
        if prof: VAL=va(prof)
        prof={}
    if rth:
        inR=True; a=int(np.floor(L[j]/BIN)); b=int(np.floor(H[j]/BIN))
        for k in range(a,b+1): prof[k]=prof.get(k,0)+V[j]/(b-a+1)
    if m>=1080 and onDay!=D[j]: onDay=D[j]; onLo=L[j]
    elif (m>=1080 or m<570) and not np.isnan(onLo): onLo=min(onLo,L[j])
    if m==570: dayLo=L[j]
    elif rth and i1<=179: dayLo=min(dayLo,L[j])
    if m>=1080 and xDay!=D[j]: xDay=D[j]; st[0]=0
    if m==570: st[1]=0
    for k in (0,1):
        if st[k]==1 and j>sig[k]: st[k]=2; xs.append(dict(k=k,date=TD[j],entry=O[j],j=j))
        if st[k]==2: st[k]=3   # 只比對進場
    if m%5==4 and (m>=1084 or m<=119) and st[0]==0 and not np.isnan(VAL) and not np.isnan(onLo):
        if onLo<VAL and C[j]>VAL:
            ok = onDay in E['va'] and E['va'][onDay]<=TS[j]; st[0]=1 if ok else 3; sig[0]=j
    ei=i1+1
    if rth and i1%5==4 and 4<=i1<=119 and not (28<=ei<=32) and st[1]==0 and not np.isnan(onLo) and not np.isnan(dayLo):
        if dayLo<onLo and C[j]>onLo:
            ok = onDay in E['on'] and E['on'][onDay]<=TS[j]; st[1]=1 if ok else 3; sig[1]=j
    if rth: pv+=(H[j]+L[j]+C[j])/3*V[j]; vv+=V[j]
    recent.append(C[j]); recent=recent[-30:]
X=pd.DataFrame(xs)
F=pd.read_csv('wr_table.csv.gz'); F['key']=F.rule+'|'+F.side
for k,key in [(0,'ASIA-VA-k1|多'),(1,'AM-ON-k1|多')]:
    a=F[(F.key==key)&(F.es_ok==True)][['date']].assign(bt=1); b=X[X.k==k][['date','entry']].assign(pine=1)
    mm=a.merge(b,on='date',how='outer'); print(key,'回測',len(a),'Pine',len(b),'兩邊都有',int((mm.bt.notna()&mm.pine.notna()).sum()),'只有 Pine',mm[mm.bt.isna()].date.tolist()[:8],'只有回測',mm[mm.pine.isna()].date.tolist()[:8])
# VWAP 判斷：前高空／前低多進場那一分鐘
nq['j']=np.arange(len(nq)); idx=nq.set_index(['trade_date','m']).j
R=F[F.key.isin(['AM-PD-k1|空','AM-PD-k1|多'])].copy()
R['m']=((R.entry_slot+18*60)%1440); R['j']=[idx.get((d,m),np.nan) for d,m in zip(R.date,R.m)]
R=R.dropna(subset=['j']); R['j']=R.j.astype(int)
R['vw']=[vwprev_at[j] for j in R.j]; R['px']=O[R.j.values]
R['pine_with']=np.where(R.dir==1,R.px>R.vw,R.px<R.vw); R['bt_with']=R.to_vwap==False
print('VWAP 順勢判斷一致',int((R.pine_with==R.bt_with).sum()),'/',len(R))
S=pd.read_csv('swan_shock.csv'); S['m']=((S.entry_slot+18*60)%1440); S['j']=[idx.get((d,m),np.nan) for d,m in zip(S.date,S.m)]
S=S.dropna(subset=['j']); S['pine']=[shock_at[int(j)] for j in S.j]
print('暴衝 ≥250 判斷一致',int(((S.pine>=250)==(S.shock>=250)).sum()),'/',len(S),'｜數值差 >1 點',int((abs(S.pine-S.shock)>1).sum()))
