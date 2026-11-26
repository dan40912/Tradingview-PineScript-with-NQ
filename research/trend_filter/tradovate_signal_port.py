import sys, numpy as np, pandas as pd
P=dict(lb=20,atrp=14,mult=0.6,minRR=1.2,minConf=82,cool=60,gcool=10,xcool=20)
COST=2.4  # pts per round trip (= $48 on NQ, project standard)

def load(tf):
    df=pd.read_csv("/home/user/Tradingview-PineScript-with-NQ/data/NQ_1m_continuous_20240102-20261019.csv.gz",
                   usecols=["timestamp_utc","open","high","low","close","volume","contract","trade_date","is_rth"])
    df["t"]=pd.to_datetime(df.timestamp_utc)
    if tf>1:
        df["g"]=df.t.dt.floor(f"{tf}min")
        df=df.groupby(["contract","g"],sort=False,as_index=False).agg(contract=("contract","first"),open=("open","first"),high=("high","max"),low=("low","min"),
            close=("close","last"),volume=("volume","sum"),trade_date=("trade_date","first"),is_rth=("is_rth","first"),t=("g","first")).reset_index(drop=True)
        df=df.sort_values("t").reset_index(drop=True)
    return df

def signals(df, vwap_mode):
    o,h,l,c,v=[df[k].to_numpy(float) for k in ["open","high","low","close","volume"]]
    td=df.trade_date.to_numpy(); n=len(c)
    typ=(h+l+c)/3; w=np.where(v>0,v,1.0)
    if vwap_mode=="cum":
        vw=np.cumsum(typ*w)/np.cumsum(w)
    else:  # session (resets at each Globex trade_date)
        s=pd.Series(typ*w).groupby(td).cumsum().to_numpy(); d=pd.Series(w).groupby(td).cumsum().to_numpy(); vw=s/d
    pc=np.r_[np.nan,c[:-1]]
    tr=np.where(np.isnan(pc),h-l,np.maximum.reduce([h-l,np.abs(h-pc),np.abs(l-pc)]))
    lb,ap=P["lb"],P["atrp"]
    res=pd.Series(h).rolling(lb).max().shift(1).to_numpy()
    sup=pd.Series(l).rolling(lb).min().shift(1).to_numpy()
    atr=pd.Series(tr).rolling(ap).mean().shift(1).to_numpy()
    avgv=pd.Series(v).rolling(lb).mean().shift(1).to_numpy()
    rng=h-l; body=np.abs(c-o); cp=np.where(rng>0,(c-l)/np.where(rng>0,rng,1),0.5)
    pvw=np.r_[np.nan,vw[:-1]]
    volx=(v>0)&(avgv>0)&(v>avgv); buf=atr*P["mult"]; overx=np.abs(c-vw)>atr*2.5
    bBO=(c>res)&(pc<=res); bV=c>vw; bRec=(pc<=pvw)&(c>vw); bSC=(cp>=0.62)&(body>=rng*0.35)
    bMaj=(bBO&bV)|(bRec&(c>=res-buf)); bPB=(l<=sup+buf)&(c>sup)&(c>o)
    bStop=np.minimum(sup,l)-buf; bRisk=c-bStop
    bTgt=c+np.maximum.reduce([bRisk*P["minRR"],atr*1.5,res-c+atr]); bRR=np.where(bRisk>0,(bTgt-c)/bRisk,0)
    sBD=(c<sup)&(pc>=sup); sV=c<vw; sRej=(pc>=pvw)&(c<vw); sSC=(cp<=0.38)&(body>=rng*0.35)
    sMaj=(sBD&sV)|(sRej&(c<=sup+buf)); sRJ=(h>=res-buf)&(c<res)&(c<o)
    sStop=np.maximum(res,h)+buf; sRisk=sStop-c
    sTgt=c-np.maximum.reduce([sRisk*P["minRR"],atr*1.5,c-sup+atr]); sRR=np.where(sRisk>0,(c-sTgt)/sRisk,0)
    def score(tr_,st,cs,rr): return 20*tr_+25*st+15*cs+10*((atr>0)&~overx)+20*rr+10*volx
    bS=score(bV|bRec,bBO|bPB,bSC,bRR>=P["minRR"]); sS=score(sV|sRej,sBD|sRJ,sSC,sRR>=P["minRR"])
    lastB=lastS=lastX=lastA=-10**9; side=None; out=[]
    for i in range(P["lb"]+1,n):
        if np.isnan(atr[i]) or np.isnan(res[i]): continue
        g=i-lastA>=P["gcool"]
        xl=side=="B" and g and i-lastX>=P["xcool"] and c[i]<vw[i] and (c[i]<sup[i] or cp[i]<=0.38)
        xs=side=="S" and g and i-lastX>=P["xcool"] and c[i]>vw[i] and (c[i]>res[i] or cp[i]>=0.62)
        cb=bS[i]>=P["minConf"] and g and bRR[i]>=P["minRR"] and bMaj[i] and (bV[i] or bRec[i]) and bSC[i] and i-lastB>=P["cool"] and side!="B"
        cs=sS[i]>=P["minConf"] and g and sRR[i]>=P["minRR"] and sMaj[i] and (sV[i] or sRej[i]) and sSC[i] and i-lastS>=P["cool"] and side!="S"
        if xl or xs:
            lastX=lastA=i; side=None; out.append((i,"X",0,0,0))
        elif cb and (not cs or bS[i]>=sS[i]):
            lastB=lastA=i; side="B"; out.append((i,"B",c[i],bStop[i],bTgt[i]))
        elif cs:
            lastS=lastA=i; side="S"; out.append((i,"S",c[i],sStop[i],sTgt[i]))
    return out

def evaluate(df, sig, maxbars, flip=False):
    h,l,c=df.high.to_numpy(),df.low.to_numpy(),df.close.to_numpy(); ct=df.contract.to_numpy()
    rows=[]
    for i,s,e,st,tg in sig:
        if s=="X": continue
        d=1 if s=="B" else -1; risk=abs(e-st); rew=abs(tg-e)
        if flip: d=-d; st,tg=e+d*-1*rew,e+d*risk  # reversed: opposite direction, stop at old target dist? keep symmetric geometry
        if flip: st,tg=e-d*risk,e+d*rew
        res=None
        for j in range(i+1,min(i+1+maxbars,len(c))):
            if ct[j]!=ct[i]: res=d*(c[j-1]-e);break
            hitS=(l[j]<=st) if d==1 else (h[j]>=st); hitT=(h[j]>=tg) if d==1 else (l[j]<=tg)
            if hitS: res=-risk;break
            if hitT: res=rew;break
        if res is None: res=d*(c[min(i+maxbars,len(c)-1)]-e)
        rows.append((df.t.iat[i],s,risk,rew,res-COST,res>0))
    return pd.DataFrame(rows,columns=["t","side","risk","rew","pnl","win"])

for tf in [5,15]:
    df=load(tf)
    for vm in ["session"]:
        sig=signals(df,vm)
        for flip in [False,True]:
            r=evaluate(df,sig,maxbars=390//tf,flip=flip)
            r["yr"]=r.t.dt.year; r["rth"]=df.set_index("t").loc[r.t,"is_rth"].to_numpy() if False else 0
            print(f"\n== {tf}m VWAP={vm} {'FLIPPED(control)' if flip else 'as signalled'}  n={len(r)}  per day={len(r)/724:.2f}")
            print(f" median risk {r.risk.median():.1f}pt  median reward {r.rew.median():.1f}pt  win {r.win.mean():.1%}  avg pnl {r.pnl.mean():+.2f}pt/trade  total {r.pnl.sum():+.0f}pt")
            print(r.groupby(["yr","side"]).pnl.agg(["count","mean"]).round(2).unstack().to_string())
            if not flip and vm=="session": r.to_csv(f"trades_{tf}m.csv",index=False)
