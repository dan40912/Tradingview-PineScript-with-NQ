import sys; sys.path.insert(0,'.')
import numpy as np
from lib import data, sim
S=sim.slot_of
g=data.load_grid("NQ",include_holdout=True)
ds=[str(d)[:10] for d in g.dates]; i=ds.index("2026-11-11")
# concatenate prev day + today for ATR continuity
O=np.concatenate([g.O[i-1],g.O[i]]);H=np.concatenate([g.H[i-1],g.H[i]]);L=np.concatenate([g.L[i-1],g.L[i]]);C=np.concatenate([g.C[i-1],g.C[i]])
off=1440
idx=np.where(np.isfinite(C))[0]
# wilder ATR on valid bars
atr=np.full(len(C),np.nan); prevc=None; a=None; n=0
for k in idx:
    tr=H[k]-L[k] if prevc is None else max(H[k]-L[k],abs(H[k]-prevc),abs(L[k]-prevc))
    n+=1
    a = tr if a is None else (a*13+tr)/14
    atr[k]=a; prevc=C[k]
def t(s): s-=off; h=(18*60+s)//60%24; m=(18*60+s)%60; return f"{h:02d}:{m:02d}"
def run(w0,w1,xend,bias):
    for k in range(off+S(w0),off+S(w1)+1):
        if np.isnan(H[k]) or np.isnan(H[k-1]) or np.isnan(H[k-2]): continue
        for d in (1,-1):
            if d==1: ok=L[k]>H[k-2]; top=L[k]; bot=H[k-2]; body=C[k-1]-O[k-1]
            else: ok=H[k]<L[k-2]; top=L[k-2]; bot=H[k]; body=O[k-1]-C[k-1]
            if not ok or top-bot<4 or body<1.5*atr[k-2]: continue
            if bias=="sweep":
                good=False
                for x in range(k-2-9,k-1):
                    if d==1 and L[x]<np.nanmin(L[x-30:x]): good=True
                    if d==-1 and H[x]>np.nanmax(H[x-30:x]): good=True
                if not good: continue
            P = top if d==1 else bot
            stop = (L[k-2]-1) if d==1 else (H[k-2]+1)
            for s in range(k+1,off+S(w1)+1):
                if (d==1 and L[s]<=P) or (d==-1 and H[s]>=P):
                    e=s+1; px=O[e]
                    if (d==1 and px<=stop) or (d==-1 and px>=stop): return f"{bias} FVG {t(k)} d{d} [{bot},{top}] touch {t(s)} entry {px} but beyond stop {stop} -> cancel"
                    sl=max(10,abs(px-stop))
                    if sl>145: return f"FVG {t(k)} sl {sl}>145 skip"
                    tp=px+d*2*sl; slp=px-d*sl
                    for q in range(e,off+S(xend)+1):
                        if np.isnan(H[q]): continue
                        if (d==1 and L[q]<=slp) or (d==-1 and H[q]>=slp): return f"{bias} FVG {t(k)} d{d} [{bot},{top}] entry {t(e)} {px} stop {slp} -> SL at {t(q)} pnl {-sl:.2f}"
                        if (d==1 and H[q]>=tp) or (d==-1 and L[q]<=tp): return f"{bias} FVG {t(k)} d{d} [{bot},{top}] entry {t(e)} {px} sl {sl:.2f} -> TP at {t(q)} pnl {2*sl:.2f}"
                    return f"{bias} FVG {t(k)} d{d} entry {px} -> time exit {C[off+S(xend)]} pnl {d*(C[off+S(xend)]-px):.2f}"
            return f"{bias} FVG {t(k)} d{d} [{bot},{top}] never touched"
    return f"{bias} no FVG"
for nm,(w0,w1,xe) in {"LON":("03:00","03:59","04:59"),"AM":("10:00","10:59","11:59"),"PM":("14:00","14:59","15:59")}.items():
    for b in ("any","sweep"): print(nm, run(w0,w1,xe,b))
