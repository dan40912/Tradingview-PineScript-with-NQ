"""Q03v2 獨立重算：σ_T、UB/LB、VWAP、追蹤出場點，隨機 40 筆逐筆比對（不用 strats 的預算陣列）。"""
import sys; sys.dont_write_bytecode=True
import numpy as np, r2lib as R, strats as S
T=S.gen_q03(1.5,"vwap_band","10:30")
rng=np.random.default_rng(3); idx=rng.choice(len(T),40,replace=False)
trad=np.where(R.g.tradable)[0]
def px(i,s):
    for k in range(s-1,s-5,-1):
        if np.isfinite(R.C[i,k]): return R.C[i,k]
    return np.nan
def o930(i):
    for k in range(930,934):
        if np.isfinite(R.O[i,k]): return R.O[i,k]
def sigma(i,s):
    prev=[j for j in trad if j<i][-14:]
    mv=[abs(px(j,s)/o930(j)-1) for j in prev]
    mv=[m for m in mv if np.isfinite(m)]
    return np.mean(mv) if len(mv)>=11 else np.nan
def vwap(i,s):
    h,l,c,v=R.H[i,930:s],R.L[i,930:s],R.C[i,930:s],R.V[i,930:s]
    m=np.isfinite(h)&np.isfinite(v)&(v>0); tp=(h+l+c)/3
    return (tp[m]*v[m]).sum()/v[m].sum()
bad=0
for k in idx:
    r=T.iloc[k]; i=r.i; s=R.slot(r['T'])
    sg=sigma(i,s); up=max(o930(i),R.PDC[i]); lo=min(o930(i),R.PDC[i])
    UB=up*(1+1.5*sg); LB=lo*(1-1.5*sg); p=px(i,s)
    ok_entry=(p>UB) if r.d==1 else (p<LB)
    # 追蹤出場：第一個 >進場的檢查點符合條件
    exp_x=None
    for c in S.CK:
        if c<=r.e: continue
        sgc=sigma(i,c); pc=px(i,c); vw=vwap(i,c)
        UBc=up*(1+1.5*sgc); LBc=lo*(1-1.5*sgc)
        if (r.d==1 and pc<max(UBc,vw)) or (r.d==-1 and pc>min(LBc,vw)):
            exp_x=c; break
    if r.why=="cond": okx = exp_x==r.x
    elif r.why=="sl": okx = exp_x is None or exp_x>r.x
    else: okx = exp_x is None
    if not (ok_entry and okx and abs(sg-r.chk_sigma)<1e-12): bad+=1; print("差異",r.date,r['T'],r.why,r.x,exp_x,sg,r.chk_sigma)
print("Q03v2 獨立重算 40 筆，差異",bad)
