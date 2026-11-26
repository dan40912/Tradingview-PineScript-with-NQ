import sys, pandas as pd, numpy as np
R="/home/user/Tradingview-PineScript-with-NQ/data/"
full=pd.read_pickle("all.pkl")
# --- daily back-adjusted closes
dc=full.groupby("trade_date").close.last(); dc.index=pd.to_datetime(dc.index)
rl=pd.read_csv(R+"NQ_ES_rollover.csv",encoding="utf-8-sig"); rl=rl[rl.symbol=="NQ"]
adj=pd.Series(0.0,index=dc.index)
for _,r in rl.iterrows(): adj[dc.index<pd.Timestamp(r.roll_date)]+=r.price_gap
c=dc+adj
s20=c.rolling(20).mean(); s10=c.rolling(10).mean()
T=pd.DataFrame({
 "T1":np.sign(c-s20),
 "T2":np.sign(c-c.shift(5)),
 "T3":np.where((c>s10)&(s10>s10.shift(5)),1,np.where((c<s10)&(s10<s10.shift(5)),-1,0)),
 "T4":np.sign(s20-s20.shift(5))},index=c.index).shift(1)   # known before the trade date
T.index=T.index.strftime("%Y-%m-%d")
def per(d):
    return np.select([d<"2025-07-01",d<"2026-01-01",d<"2026-10-01",d<="2026-11-24"],["A 24/01-25/06","B 25/07-12","C 26/01-09","D 26/10-11/24"],"")
def report(df,lab,needpos=False):
    df=df[df.per!=""].join(T,on="date")
    out=[]
    for f in ["無過濾","T1","T2","T3","T4"]:
        x=df if f=="無過濾" else df[(df[f]==df.dir)|((f=="T3")&(df[f]==0))]
        g=x.groupby("per").pnl.agg(n="size",win=lambda s:(s>0).mean(),avg="mean",tot="sum")
        g["f"]=f; out.append(g.reset_index())
    o=pd.concat(out); base=o[o.f=="無過濾"].set_index("per")
    print(f"\n######## {lab}")
    for f in ["無過濾","T1","T2","T3","T4"]:
        r=o[o.f==f].set_index("per"); cells=[]
        ok=True
        for p in r.index:
            cells.append(f"{int(r.n[p]):4d}筆 {r.win[p]:.0%} {r.avg[p]:+6.0f}/筆")
            if p[0] in "ABC" and f!="無過濾":
                ok&= r.avg[p]>base.avg[p] and r.win[p]>=base.win[p] and r.n[p]>=0.5*base.n[p]*(0 if needpos else 1)
                if needpos: ok&= r.avg[p]>0
        print(f"{f:4s} | "+" | ".join(cells)+("" if f=="無過濾" else ("   → 通過" if ok else "   → 沒通過")))
    print("      欄位順序:", " | ".join(o.per.unique()))
    return df
# --- v18 plan
sys.argv=["x","tv/nq_full_tr.csv","tv/es_full_tr.csv"]
ns={}; exec(open("tv/book6.py").read().split("if __name__")[0],ns)
plan=ns["build"](ns["n"]); plan["per"]=per(plan.date)
v=report(plan[["date","dir","pnl","per","kind","mae"]].copy(),"v18 第一層計畫（1 口 NQ）")
v.to_csv("v18_full_with_trend.csv",index=False)
# by kind under best filter shown later
# --- pasted indicator
g={}; exec(open("tradovate_signal_port.py").read().split("for tf in")[0],g)
def rs(df,tf):
    if tf==1: return df.reset_index(drop=True)
    df=df.copy(); df["g"]=df.t.dt.floor(f"{tf}min")
    return df.groupby(["contract","g"],as_index=False,sort=False).agg(open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum"),trade_date=("trade_date","first"),is_rth=("is_rth","first"),t=("g","first")).sort_values("t").reset_index(drop=True)
for tf in [1,5]:
    df=rs(full,tf); sig=g["signals"](df,"session")
    h,l,cc,td,ct=df.high.to_numpy(),df.low.to_numpy(),df.close.to_numpy(),df.trade_date.to_numpy(),df.contract.to_numpy()
    rows=[]
    for i,s,e,st,tg in sig:
        if s=="X": continue
        d=1 if s=="B" else -1; risk=abs(e-st); rew=abs(tg-e); out=None
        for j in range(i+1,min(i+1+3*390//tf,len(cc))):
            if ct[j]!=ct[i]: out=d*(cc[j-1]-e);break
            if (l[j]<=st if d==1 else h[j]>=st): out=-risk;break
            if (h[j]>=tg if d==1 else l[j]<=tg): out=rew;break
        if out is None: out=d*(cc[j]-e)
        rows.append((td[i],d,out*20-48))
    r=pd.DataFrame(rows,columns=["date","dir","pnl"]); r["per"]=per(r.date)
    report(r,f"貼來的 Tradovate 指標 {tf} 分K（1 口 NQ）",needpos=True).to_csv(f"trad_{tf}m_with_trend.csv",index=False)
