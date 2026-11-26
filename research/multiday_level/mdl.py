import pandas as pd, numpy as np, itertools
a=pd.read_csv("tv/nq_full.csv.gz"); b=pd.read_csv("tv/nq.csv.gz")
d=pd.concat([a,b]).drop_duplicates("timestamp_utc").sort_values("timestamp_utc").reset_index(drop=True)
d["contract"]=d.contract.fillna("NQZ26")
d["t"]=pd.to_datetime(d.timestamp_utc); et=d.t.dt.tz_convert("America/New_York")
d["td"]=(et+pd.Timedelta(hours=6)).dt.strftime("%Y-%m-%d"); d["m"]=et.dt.hour*60+et.dt.minute
days=[]
for td,x in d.groupby("td",sort=True):
    r=x[(x.m>=570)&(x.m<960)]
    if len(r)<300: continue
    days.append(dict(td=td,lo=x.low.min(),hi=x.high.max(),rr=r.high.max()-r.low.min(),ct=x.contract.iloc[-1],x=r))
def per(td):
    return "A" if td<"2025-07-01" else "B" if td<"2026-01-01" else "C" if td<"2026-10-01" else "D"
res=[]
for k in range(10,len(days)):
    D=days[k]; adr=np.mean([q["rr"] for q in days[k-10:k]]); x=D["x"]
    o,h,l,c,m=[x[f].to_numpy() for f in ["open","high","low","close","m"]]
    # 5m bars on clock
    b5=(m-570)//5
    for N in (3,5):
        prev=days[k-N:k]
        if any(q["ct"]!=D["ct"] for q in prev): continue
        L=min(q["lo"] for q in prev); H=max(q["hi"] for q in prev)
        for z,trig,W,ex in itertools.product((0.05,0.10),("touch","sweep"),(690,900),("E1","E2")):
            for side in (1,-1):
                lev=L if side==1 else H
                entry_i=None
                for bi in np.unique(b5):
                    idx=np.where(b5==bi)[0]
                    end_m=570+5*(bi+1)        # entry minute
                    if end_m<575 or end_m>W: continue
                    bo,bh,bl,bc=o[idx[0]],h[idx].max(),l[idx].min(),c[idx[-1]]
                    if side==1:
                        ok=(bl<=lev+z*adr if trig=="touch" else bl<lev) and bc>lev and (bc>bo if trig=="touch" else True)
                    else:
                        ok=(bh>=lev-z*adr if trig=="touch" else bh>lev) and bc<lev and (bc<bo if trig=="touch" else True)
                    if ok:
                        j=np.where(m==end_m)[0]
                        if len(j): entry_i=j[0]
                        break
                if entry_i is None: continue
                e=o[entry_i]
                if ex=="E1": sl,tp,tend=e-side*145,e+side*55,(749 if W==690 else 955)
                else:
                    sl=lev-side*0.05*adr; risk=side*(e-sl)
                    if risk<=0: continue
                    tp,tend=e+side*2*risk,955
                pnl=None
                for jj in range(entry_i,len(m)):
                    if m[jj]>tend: pnl=side*(c[jj-1]-e);break
                    hs=(l[jj]<=sl) if side==1 else (h[jj]>=sl); ht=(h[jj]>=tp) if side==1 else (l[jj]<=tp)
                    if hs: pnl=side*(sl-e);break
                    if ht: pnl=side*(tp-e);break
                if pnl is None: pnl=side*(c[-1]-e)
                res.append(dict(td=D["td"],per=per(D["td"]),N=N,z=z,trig=trig,W=W,ex=ex,side=side,lev=lev,entry=e,et=int(m[entry_i]),risk=abs(e-sl),pnl=pnl*20-48))
r=pd.DataFrame(res); r.to_csv("mdl_trades.csv",index=False); print(len(r))
