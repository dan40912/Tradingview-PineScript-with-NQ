"""高波動的調整：I2 v1 和參考點（前高空／前低多）在四種做法下的結果。
基準 55/145×1 口；VS＝止盈止損乘上 ATR10/REF；SZ＝口數＝floor(10/倍數)/10（MNQ 為單位，最少 0.2、最多 1 口 NQ）；VS+SZ＝兩個一起。
REF＝2024–2025 的 ATR10 中位數（255.2 點），ATR10＝前 10 個交易日美股盤中振幅平均（開盤前就知道）。
執行方式和 forward.py 相同（需要 dazzling-euler 分支的 research/lib）。研究用途，不是投資建議。"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from lib import data, features, sim, strategies
from lib.data import RTH_OFF
g=data.load_grid('NQ',include_holdout=True); lv=features.daily_levels(g)
# --- daily RTH range and ATR10 (prior 10 tradable days, known before the open)
H=g.rth(g.H); L=g.rth(g.L)
rng=pd.Series(np.nanmax(H,1)-np.nanmin(L,1),index=g.dates)
tr=rng[g.tradable]
atr10=tr.rolling(10).mean().shift(1).reindex(g.dates).ffill()
REF=float(atr10[(atr10.index>='2024-01-17')&(atr10.index<='2025-12-31')].median())
scale=(atr10/REF)
print('REF ATR10 (2024-25 median) =',round(REF,1))
def ref_signals(g,lv):
    sigs=[]
    for r in lv.itertuples():
        if not r.has_prev or not g.tradable[r.i] or np.isnan(r.open_0930): continue
        i=r.i; Hh=g.H[i]; Ll=g.L[i]; C=g.C[i]; st={0:0,1:0}
        for idx in range(4,120,5):
            c=C[RTH_OFF+idx]
            if np.isnan(c): continue
            dh=np.nanmax(Hh[RTH_OFF:RTH_OFF+idx+1]); dl=np.nanmin(Ll[RTH_OFF:RTH_OFF+idx+1])
            ei=idx+1; okE=5<=ei<=120 and not (28<=ei<=32)
            if st[0]==0 and dh>r.pdh and c<r.pdh and okE:
                st[0]=1; sigs.append({'i':i,'entry_slot':RTH_OFF+ei,'dir':-1,'tag':'前高空'})
            if st[1]==0 and dl<r.pdl and c>r.pdl and okE:
                st[1]=1; sigs.append({'i':i,'entry_slot':RTH_OFF+ei,'dir':1,'tag':'前低多'})
    return sigs
base_sigs={'I2':[dict(s,tag='I2') for s in strategies.i2_v1_signals(g,lv)],'REF':ref_signals(g,lv)}
# reproduce check vs repo CSV
rc=pd.read_csv('/home/user/Tradingview-PineScript-with-NQ/backtest/reference_points_trades.csv',encoding='utf-8-sig')
rc=rc[rc.rule.str.contains('前高|前低')]
ours=sim.run_signals(g,base_sigs['REF'],exit_slot=RTH_OFF+179)
ours=ours[ours.date<='2026-10-27']
rc['k']=rc.date+rc.direction.map({'long':'1','short':'-1'}); ours['k']=ours.date+ours.dir.astype(str)
m=rc.merge(ours,on='k',how='outer',indicator=True)
print('ref-point reproduce:',m._merge.value_counts().to_dict(),'pnl diff>1:',((m.net_usd-m.pnl).abs()>1).sum())
def period(d):
    return np.select([d>='2026-10-01',d>='2026-01-01',d>='2025-01-01'],['4) 2026-10~11/10','3) 2026-01~09','2) 2025'],'1) 2024')
def run(variant):
    out=[]
    for fam,sigs in base_sigs.items():
        ss=[]
        for s in sigs:
            s=dict(s); d=g.dates[s['i']]; k=scale[d]
            if np.isnan(k): continue
            tp,sl,ct=55.0,145.0,1.0
            if variant in('VS','VS+SZ'): tp=round(55*k*4)/4; sl=round(145*k*4)/4
            if variant in('SZ','VS+SZ'): ct=float(np.clip(np.floor(10/k)/10,0.2,1.0))
            s.update(tp_pts=tp,sl_pts=sl,contracts=ct,exit_slot=RTH_OFF+179,fam=fam,k=k); ss.append(s)
        out.append(sim.run_signals(g,ss))
    t=pd.concat(out).sort_values(['date','entry_slot']); t['p']=period(t.date); t['var']=variant; return t
def dd(x):
    c=x.cumsum(); return (c-c.cummax()).min()
res=[]
allt=[]
for v in ['基準 55/145×1口','VS','SZ','VS+SZ']:
    t=run(v if v!='基準 55/145×1口' else 'base'); t['var']=v; allt.append(t)
    for (p,fam),x in list(t.groupby(['p','fam']))+[((p,'合併'),x) for p,x in t.groupby('p')]:
        res.append(dict(var=v,period=p,fam=fam,n=len(x),total=x.pnl.sum(),avg=x.pnl.mean(),maxDD=dd(x.pnl),worst=x.pnl.min()))
R=pd.DataFrame(res).round(0)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
for fam in ['I2','REF','合併']:
    print('=====',fam); print(R[R.fam==fam].pivot_table(index='var',columns='period',values=['total','maxDD','worst'],sort=False).round(0).to_string())
A=pd.concat(allt); A.to_pickle('volopt_trades.pkl')
print(scale[scale.index>='2026-09-25'].round(2).to_string())
