"""Q1_critic：NWOG/NDOG 觸及 CE 的天數、S06 滾動 SMT 頻率（只數，不回測）。研究用途。"""
import sys; sys.path.insert(0, '/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/wt/research')
import numpy as np, pandas as pd
from lib import data, features
g = data.load_grid('NQ', include_holdout=True); lv = features.daily_levels(g); e = data.load_grid('ES', include_holdout=True)
D = pd.to_datetime(pd.Series(g.dates)); n=len(D)
def last_close(i):
    c=g.C[i,1300:1380]; c=c[np.isfinite(c)]; return c[-1] if len(c) else np.nan
def first_open(i):
    o=g.O[i,0:15]; o=o[np.isfinite(o)]; return o[0] if len(o) else np.nan
res={'NDOG':[0,0,0],'NWOG':[0,0,0]}
cur=None
for i in range(1,n):
    if not (lv.tradable.iloc[i] and lv.has_prev.iloc[i]): 
        if D[i].weekday()==0: cur=None
        continue
    same = g.contract[i]==g.contract[i-1]
    pc, op = last_close(i-1), first_open(i)
    nd = (min(pc,op), max(pc,op)) if same else None
    if D[i].weekday()==0 or (D[i]-D[i-1]).days>=3: cur = nd
    for name,gap in (('NDOG',nd),('NWOG',cur)):
        if gap is None or not np.isfinite(gap[0]): continue
        lo,hi=gap; ce=(lo+hi)/2; res[name][0]+=1
        if hi-lo<5: continue
        res[name][1]+=1
        o=g.O[i,930]; L=np.nanmin(g.L[i,930:1080]); H=np.nanmax(g.H[i,930:1080])
        if (o>hi and L<=ce) or (o<lo and H>=ce): res[name][2]+=1
print('天數／缺口>=5／開在外且 09:30-12:00 觸 CE：', res)
# S06：滾動 LB 的 SMT，09:30-11:00 第一個訊號出現的時間分佈
def bars5(G,i):
    a=np.arange(930,1320,5); 
    H=np.array([np.nanmax(G.H[i,s:s+5]) if np.isfinite(G.H[i,s:s+5]).any() else np.nan for s in a])
    L=np.array([np.nanmin(G.L[i,s:s+5]) if np.isfinite(G.L[i,s:s+5]).any() else np.nan for s in a])
    return H,L
eidx={d:k for k,d in enumerate(e.dates)}
first=[]; cnt=[]
for i in range(n):
    if not lv.tradable.iloc[i] or g.dates[i] not in eidx: continue
    j=eidx[g.dates[i]]
    # 含 09:30 前 24 根
    a=np.arange(930-120,1320,5)
    def b(G,k):
        H=np.array([np.nanmax(G.H[k,s:s+5]) if np.isfinite(G.H[k,s:s+5]).any() else np.nan for s in a])
        L=np.array([np.nanmin(G.L[k,s:s+5]) if np.isfinite(G.L[k,s:s+5]).any() else np.nan for s in a])
        return H,L
    nh,nl=b(g,i); eh,el=b(e,j); LB=12; c=0; f=None
    for k in range(24, 24+18):   # 09:30-10:59
        if k<LB: continue
        x=[nl[k],el[k]]; 
        if np.isnan(x).any(): continue
        nq_new = nl[k]<np.nanmin(nl[k-LB:k]); es_new = el[k]<np.nanmin(el[k-LB:k])
        nqh = nh[k]>np.nanmax(nh[k-LB:k]); esh = eh[k]>np.nanmax(eh[k-LB:k])
        if nq_new!=es_new or nqh!=esh:
            c+=1; f = k if f is None else f
    cnt.append(c); first.append(f)
first=np.array([x for x in first if x is not None]); 
print('S06 LB=12：AM 窗每天未確認 SMT 根數 平均', np.mean(cnt), '；有的天比例', np.mean(np.array(cnt)>0))
print('第一個 SMT 的時間（距 09:30 分鐘）分位 25/50/75：', np.percentile((first-24)*5,[25,50,75]))
