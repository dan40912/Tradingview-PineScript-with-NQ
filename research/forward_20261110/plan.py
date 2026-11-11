"""一天多個進場點的組合：亞洲盤反向（A1）＋I2＋前高空＋前低多，每筆 5 口 MNQ（0.5 口 NQ），+55／−145。
執行方式和 forward.py 相同（需要 dazzling-euler 分支的 research/lib）。
每日停損：已實現虧損 ≤ −$2,900 後不再進新單（依進場時間排序）。研究用途，不是投資建議。"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from lib import data, features, sim, strategies
from lib.data import RTH_OFF
from lib.sim import hhmm_of
g=data.load_grid('NQ',include_holdout=True); lv=features.daily_levels(g)
def ref_signals():
    out=[]
    for r in lv.itertuples():
        if not r.has_prev or not g.tradable[r.i] or np.isnan(r.open_0930): continue
        i=r.i; st={0:0,1:0}
        for idx in range(4,120,5):
            c=g.C[i][RTH_OFF+idx]
            if np.isnan(c): continue
            dh=np.nanmax(g.H[i][RTH_OFF:RTH_OFF+idx+1]); dl=np.nanmin(g.L[i][RTH_OFF:RTH_OFF+idx+1])
            ei=idx+1; okE=5<=ei<=120 and not (28<=ei<=32)
            if st[0]==0 and dh>r.pdh and c<r.pdh and okE: st[0]=1; out.append(dict(i=i,entry_slot=RTH_OFF+ei,dir=-1,tag='前高空'))
            if st[1]==0 and dl<r.pdl and c>r.pdl and okE: st[1]=1; out.append(dict(i=i,entry_slot=RTH_OFF+ei,dir=1,tag='前低多'))
    return out
def a1_signals():
    out=[]
    for r in lv.itertuples():
        if not r.has_prev or np.isnan(r.pdh): continue
        i=r.i; H,L,C=g.H[i],g.L[i],g.C[i]
        if np.isfinite(H[:540]).sum()<400: continue
        swh=swl=False; ds=dl_=False
        for k in range(0,108):
            a=5*k; b=a+5; ok=~np.isnan(C[a:b])
            if not ok.any(): continue
            last=a+np.where(ok)[0][-1]
            if last>480: break
            if np.nanmax(H[a:b])>r.pdh: swh=True
            if np.nanmin(L[a:b])<r.pdl: swl=True
            c=C[last]; e=last+1
            if swh and not ds and c<r.pdh: out.append(dict(i=i,entry_slot=e,dir=-1,tag='亞洲反向空',exit_slot=539)); ds=True
            if swl and not dl_ and c>r.pdl: out.append(dict(i=i,entry_slot=e,dir=1,tag='亞洲反向多',exit_slot=539)); dl_=True
    return out
S=[dict(s,tag='I2') for s in strategies.i2_v1_signals(g,lv)]+ref_signals()+a1_signals()
T=sim.run_signals(g,S,tp_pts=55.0,sl_pts=145.0,exit_slot=RTH_OFF+179,contracts=0.5)
T=T.sort_values(['date','entry_slot']).reset_index(drop=True)
keep=[]
for d,x in T.groupby('date',sort=False):
    closed=[]
    for t in x.itertuples():
        real=sum(p for xs,p in closed if xs<=t.entry_slot)
        if real<=-2900: keep.append(False); continue
        keep.append(True); closed.append((t.exit_slot,t.pnl))
T['taken']=keep
T.to_csv('plan_trades.csv',index=False)
P=lambda d: np.select([d>='2026-10-20',d>='2026-01-01',d>='2025-01-01'],['4) 10/20~11/10','3) 2026-01~10/19','2) 2025'],'1) 2024')
T['P']=P(T.date)
t=T[T.taken]
def dd(x): c=x.cumsum(); return (c-c.cummax()).min()
nd=pd.Series(g.dates[g.tradable]); nd=nd[nd>='2024-01-03']; ndays=nd.groupby(P(nd)).size()
pd.set_option('display.width',250)
rows=[]
for p,x in t.groupby('P'):
    dy=x.groupby('date').pnl.sum()
    rows.append(dict(P=p,trading_days=ndays[p],trades=len(x),per_day=len(x)/ndays[p],days_ge2=(x.groupby('date').size()>=2).sum(),win=(x.pnl>0).mean()*100,total=x.pnl.sum(),avg=x.pnl.mean(),maxDD=dd(x.pnl),worst_day=dy.min(),best_day=dy.max(),neg_days=(dy<0).mean()*100))
print(pd.DataFrame(rows).round(1).to_string(index=False))
print(t.groupby(['tag','P']).pnl.agg(['size','sum','mean']).round(0).unstack('P').to_string())
print(T[T.date>='2026-11-02'][['date','tag','dir','entry_time','entry','exit_time','reason','pnl','taken']].to_string(index=False))
