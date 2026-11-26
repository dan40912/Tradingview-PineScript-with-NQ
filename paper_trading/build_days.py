import sys, json, math, numpy as np, pandas as pd
rng=np.random.default_rng(20261126)
d=pd.read_pickle("full_clean.pkl")
b=pd.read_csv("tv/nq.csv.gz"); b["t"]=pd.to_datetime(b.timestamp_utc); et=b.t.dt.tz_convert("America/New_York")
b["td"]=(et+pd.Timedelta(hours=6)).dt.strftime("%Y-%m-%d"); b["m"]=et.dt.hour*60+et.dt.minute; b["contract"]="NQZ26"
d=pd.concat([d,b[b.t>d.t.max()]]).sort_values("t").reset_index(drop=True)
vm=pd.read_csv("daily_vm.csv",index_col=0)
ROLL={'2024-03-12','2024-06-18','2024-09-17','2024-12-17','2025-03-18','2025-06-17','2025-09-16','2025-12-16','2026-03-17','2026-06-16','2026-09-15'}
def value_area(x,BIN=5.0,FR=0.7):
    prof={}
    for lo,hi,v in zip(x.low.values,x.high.values,x.volume.values):
        a=int(math.floor(lo/BIN)); bb=int(math.floor(hi/BIN)); nb=bb-a+1
        for k in range(a,bb+1): prof[k]=prof.get(k,0.0)+v/nb
    keys=sorted(prof); kmin,kmax=keys[0],keys[-1]; tot=sum(prof.values()); poc,pv=kmin,-1
    for k in keys:
        if prof[k]>pv: poc,pv=k,prof[k]
    lo=hi=poc; acc=pv
    while acc<FR*tot and (lo>kmin or hi<kmax):
        lv=prof.get(lo-1,-1.0); hv=prof.get(hi+1,-1.0)
        if hv>lv: hi+=1; acc+=max(0,hv)
        else: lo-=1; acc+=max(0,lv)
    return lo*BIN,poc*BIN,(hi+1)*BIN
G={td:x for td,x in d.groupby("td",sort=True)}
tds=[t for t in G if t in vm.index and not np.isnan(vm.loc[t,"adr10"])]
# v18 signals
sys.argv=["x","tv/nq_full_tr.csv","tv/es_full_tr.csv"]; ns={}; exec(open("tv/book6.py").read().split("if __name__")[0],ns)
nA=ns["n"]; pA=ns["build"](nA)
sys.argv=["x","tv/nq_tr.csv","tv/es_tr.csv"]; ns2={}; exec(open("tv/book6.py").read().split("if __name__")[0],ns2)
nB=ns2["n"]; pB=ns2["build"](nB)
n=pd.concat([nA[nA.date<"2026-11-25"],nB[nB.date=="2026-11-25"]]); p=pd.concat([pA[pA.date<"2026-11-25"],pB[pB.date=="2026-11-25"]])
pkey=set(zip(p.date,p.kind,p.et.astype(str)))
ok=[]
for i,t in enumerate(tds):
    if i<6 or t in ROLL: continue
    x=G[t]; r=x[(x.m>=420)&(x.m<960)]
    if len(r)<480 or x.contract.iloc[-1]!=G[tds[i-1]].contract.iloc[-1]: continue
    ok.append(t)
hv=[t for t in ok if vm.loc[t,"vm"]>=1.8]; nv=[t for t in ok if vm.loc[t,"vm"]<1.8]
pick=sorted(set(rng.choice(hv,40,replace=False))|set(rng.choice(nv,80,replace=False)))
days=[]
for t in pick:
    i=tds.index(t); x=G[t]; prev=G[tds[i-1]]
    pr=prev[(prev.m>=570)&(prev.m<960)]
    val,poc,vah=value_area(pr)
    on=x[(x.m<570)|(x.m>=1080)]
    lows=[G[tds[j]].low.min() for j in range(i-5,i)]; highs=[G[tds[j]].high.max() for j in range(i-5,i)]
    r=x[(x.m>=420)&(x.m<960)]
    base=float(r.open.iloc[0])
    bars=[[int(mm-420),*[int(round((v-base)*4)) for v in (o,h,l,c)],int(vv)] for mm,o,h,l,c,vv in zip(r.m,r.open,r.high,r.low,r.close,r.volume)]
    s=n[n.date==t]; sig=[]
    for _,q in s.iterrows():
        e=q.et.tz_convert("America/New_York"); mm=e.hour*60+e.minute
        if e.strftime("%Y-%m-%d")!=t and mm>=1080: mm=-1   # 前一晚亞洲盤
        xt=q.xt.tz_convert("America/New_York") if pd.notna(q.xt) else None
        inplan=(t,q.kind,str(q.et)) in pkey
        tier1=q.tier==1 or inplan
        sig.append(dict(m=int(mm),dir=int(q.dir),kind=str(q.kind),tier=int(q.tier),plan=bool(inplan),
            skip=("" if pd.isna(q.skipWhy) else str(q.skipWhy).strip('"')),entry=float(q.entry),
            tp=float(q.entry+q.dir*55) if tier1 else None,sl=float(q.entry-q.dir*145) if tier1 else None,
            xm=(xt.hour*60+xt.minute) if xt is not None else None,exit=None if pd.isna(q.exit) else float(q.exit),why=str(q.why),pnl=None if pd.isna(q.pnl) else float(q.pnl)))
    days.append(dict(td=t,base=base,vm=round(float(vm.loc[t,"vm"]),2),adr=round(float(vm.loc[t,"adr10"]),1),
        lv=dict(VAH=vah,POC=poc,VAL=val,PDH=float(pr.high.max()),PDL=float(pr.low.min()),PDC=float(pr.close.iloc[-1]),
                ONH=float(on.high.max()),ONL=float(on.low.min()),H3=float(max(highs[-3:])),L3=float(min(lows[-3:])),H5=float(max(highs)),L5=float(min(lows))),
        bars=bars,sig=sig))
json.dump(days,open("paper/days.json","w"),separators=(",",":"))
import os; print(len(days),"days",os.path.getsize("paper/days.json")//1024,"KB; hv",sum(1 for q in days if q["vm"]>=1.8),"| signals",sum(len(q["sig"]) for q in days),"plan",sum(s["plan"] for q in days for s in q["sig"]))
print([q["td"] for q in days][-8:])
