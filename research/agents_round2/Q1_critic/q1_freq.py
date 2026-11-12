"""Q1_critic 快速頻率檢查（不是回測，只數訊號／停損距離分佈）。研究用途。"""
import sys; sys.path.insert(0, '/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/wt/research')
import numpy as np, pandas as pd
from lib import data, features
g = data.load_grid('NQ', include_holdout=True); lv = features.daily_levels(g)
D = g.dates; n = len(D)
rth = [(i) for i in range(n) if lv.tradable.iloc[i] and lv.has_prev.iloc[i]]
print('可交易日', len(rth))
# 換月日
roll = [i for i in range(1, n) if g.contract[i] != g.contract[i-1]]
print('換月日數', len(roll), [D[i] for i in roll][:12])
# RTH 高低
RH = np.array([np.nanmax(g.H[i,930:1320]) if np.isfinite(g.H[i,930:1320]).any() else np.nan for i in range(n)])
RL = np.array([np.nanmin(g.L[i,930:1320]) if np.isfinite(g.L[i,930:1320]).any() else np.nan for i in range(n)])
ADR = pd.Series(RH-RL).rolling(10).mean().shift(1).values
# S11: N 日低/高掃 + 收回 (09:30-12:00)，只算同合約
for N in (1,5,20):
    cnt=0; cnt_ok=0
    for i in rth:
        if i < N: continue
        js = list(range(i-N, i))
        if any(g.contract[j]!=g.contract[i] for j in js): continue
        lo = np.nanmin(RL[js]); hi = np.nanmax(RH[js])
        L = g.L[i,930:1080]; H = g.H[i,930:1080]; C = g.C[i,930:1080]
        hitL = np.where(L <= lo-3)[0]; hitH = np.where(H >= hi+3)[0]
        f=False
        if len(hitL):
            k=hitL[0]; r=np.where(C[k:]>=lo+0.5)[0]
            if len(r): f=True
        if len(hitH):
            k=hitH[0]; r=np.where(C[k:]<=hi-0.5)[0]
            if len(r): f=True
        cnt += f; cnt_ok += 1
    print(f'S11 N={N}: 有訊號天數 {cnt}/{cnt_ok}')
# S12 OR stop > 145
for T in (5,15,30):
    w=[]; 
    for i in rth:
        h=np.nanmax(g.H[i,930:930+T]); l=np.nanmin(g.L[i,930:930+T]); w.append(h-l)
    w=np.array(w); print(f'S12 T={T}: OR 寬中位 {np.nanmedian(w):.0f}，>145 比例 {(w>145).mean():.0%}')
# S13 IBR
ib=np.array([np.nanmax(g.H[i,930:990])-np.nanmin(g.L[i,930:990]) for i in rth])
print(f'S13 IBR 中位 {np.median(ib):.0f}；0.25×IBR 停損 >145 比例 {(0.25*ib>145).mean():.0%}；IBR<=0.35ADR 比例 {np.mean(ib<=0.35*ADR[rth]):.0%}')
# S05 NDOG 大小
nd=[]; 
for i in rth:
    if i==0 or g.contract[i]!=g.contract[i-1]: continue
    pc = g.C[i-1, 1379] if np.isfinite(g.C[i-1,1379]) else np.nan
    if np.isnan(pc):
        c = g.C[i-1, 1300:1380]; c=c[np.isfinite(c)]; pc = c[-1] if len(c) else np.nan
    o = g.O[i,0:15]; o=o[np.isfinite(o)]
    if len(o) and np.isfinite(pc): nd.append(abs(o[0]-pc))
nd=np.array(nd); print(f'S05 NDOG 大小中位 {np.median(nd):.1f}；>=5 {np.mean(nd>=5):.0%}；>=20 {np.mean(nd>=20):.0%}')
# S01 1分 FVG（disp 1.5 ATR14_1m）在 AM 窗的數量
def atr1(i):
    H,L,C=g.H[i],g.L[i],g.C[i]; pc=np.r_[np.nan,C[:-1]]
    tr=np.nanmax(np.vstack([H-L,np.abs(H-pc),np.abs(L-pc)]),axis=0)
    s=pd.Series(tr).ffill(); return s.ewm(alpha=1/14,adjust=False).mean().values
cntw={'AM':0,'PM':0}; days_any={'AM':0,'PM':0}
for i in rth[::3]:
    A=atr1(i); O,H,L,C=g.O[i],g.H[i],g.L[i],g.C[i]
    for w,(a,b) in {'AM':(960,1020),'PM':(1200,1260)}.items():
        k_ok=0
        for k in range(a,b):
            if not np.isfinite([H[k-2],L[k],O[k-1],C[k-1]]).all(): continue
            body=C[k-1]-O[k-1]
            if L[k]-H[k-2]>=4 and body>=1.5*A[k-2]: k_ok+=1
            if L[k-2]-H[k]>=4 and -body>=1.5*A[k-2]: k_ok+=1
        cntw[w]+=k_ok; days_any[w]+= k_ok>0
m=len(rth[::3]); print(f'S01 每窗合格 FVG 平均數 AM {cntw["AM"]/m:.2f} PM {cntw["PM"]/m:.2f}；有合格 FVG 的日比例 AM {days_any["AM"]/m:.0%} PM {days_any["PM"]/m:.0%}')
# S16 gap 條件比例
gp=[]
for i in rth:
    if g.contract[i]!=g.contract[i-1]: continue
    gp.append((lv.open_0930.iloc[i]-lv.pdc.iloc[i])/ADR[i] if np.isfinite(ADR[i]) else np.nan)
gp=np.abs(np.array(gp)); print(f'S16 |gap|/ADR 在 0.10～0.25 {np.nanmean((gp>=0.1)&(gp<=0.25)):.0%}，0.10～0.5 {np.nanmean((gp>=0.1)&(gp<=0.5)):.0%}')
# S10 60 分區間 R<=0.35ADR 且 >=20 在 10:00-14:30 的比例（每 5 分取樣）
ok=0; tot=0
for i in rth[::3]:
    for a in range(1000-1000%5, 1230, 5):
        h=np.nanmax(g.H[i,a-60:a]); l=np.nanmin(g.L[i,a-60:a]); R=h-l; tot+=1; ok+= (R<=0.35*ADR[i]) and R>=20
print(f'S10 盤整條件成立比例（5 分 K 取樣）{ok/tot:.0%}')
