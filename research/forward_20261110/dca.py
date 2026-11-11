"""執行方式和 forward.py 相同（需要 dazzling-euler 分支的 research/lib）。
DCA（浮虧加碼）與順勢加碼，用 I2 v1 的進場點，MNQ 計價。
預算：每天最多虧 $3,000（含已實現），獲利目標 $1,100（整筆部位合計）。12:29 時間到出場。
MNQ 每點 $2，成本每口來回 $4.8（＝NQ $48 的 1/10）。
同一分鐘：先處理不利方向（加碼成交、停損），加碼成交的那一分鐘不算止盈（保守）。研究用途，不是投資建議。"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from lib import data, features, strategies
from lib.data import RTH_OFF
PT=2.0; COST=4.8; LOSS=3000.0; TARGET=1100.0
g=data.load_grid('NQ',include_holdout=True); lv=features.daily_levels(g).set_index('i')
sigs=strategies.i2_v1_signals(g,lv.reset_index())
def run(sig,plan):
    """plan: list of (qty, trigger) ; trigger=None → at entry; ('adv',X) adverse X pts; ('fav',X) favourable X pts;
       ('lvl','VA') add at VAL(long)/VAH(short); ('lvl','EXT') add at day low(long)/high(short) so far."""
    i=sig['i']; d=sig['dir']; e=sig['entry_slot']; xs=RTH_OFF+179
    O,H,L,C=g.O[i],g.H[i],g.L[i],g.C[i]
    while np.isnan(O[e]): e+=1
    px0=O[e]; r=lv.loc[i]
    ext=(np.nanmin(L[RTH_OFF:e]) if d==1 else np.nanmax(H[RTH_OFF:e]))
    lvl={'VA':(r.val if d==1 else r.vah),'EXT':ext}
    pos=[]; pending=[]
    for q,t in plan:
        if t is None: pos.append((px0,q))
        elif t[0]=='adv': pending.append((q,px0-d*t[1],'adv'))
        elif t[0]=='fav': pending.append((q,px0+d*t[1],'fav'))
        elif t[0]=='lvl':
            p=lvl[t[1]]
            if (d==1 and p<px0) or (d==-1 and p>px0): pending.append((q,p,'adv'))
    def pnl(p): return sum(q*(p-a)*d for a,q in pos)*PT
    def solve(target):
        Q=sum(q for _,q in pos); return (target/PT/d + sum(q*a for a,q in pos))/Q if Q else np.nan
    for s in range(e,xs+1):
        h,l=H[s],L[s]
        if np.isnan(h): continue
        worst=l if d==1 else h; best=h if d==1 else l
        added=False
        for k,(q,p,kind) in enumerate(list(pending)):
            if kind=='adv' and ((d==1 and worst<=p) or (d==-1 and worst>=p)):
                pos.append((p,q)); pending.remove((q,p,kind)); added=True
        for k,(q,p,kind) in enumerate(list(pending)):
            if kind=='fav' and ((d==1 and best>=p) or (d==-1 and best<=p)):
                pos.append((p,q)); pending.remove((q,p,kind)); added=True
        if pnl(worst)<=-LOSS:
            xp=solve(-LOSS); return dict(reason='sl',gross=-LOSS,qty=sum(q for _,q in pos),slot=s)
        if not added and pnl(best)>=TARGET:
            return dict(reason='tp',gross=TARGET,qty=sum(q for _,q in pos),slot=s)
    cc=C[e:xs+1]; last=cc[~np.isnan(cc)][-1]
    return dict(reason='time',gross=pnl(last),qty=sum(q for _,q in pos),slot=xs)
PLANS={
 '基準：10 口一次進（＝I2 v1）':[(10,None)],
 'DCA 5+5，逆向 30 點加':[(5,None),(5,('adv',30))],
 'DCA 5+5，逆向 50 點加':[(5,None),(5,('adv',50))],
 'DCA 5+5，逆向 75 點加':[(5,None),(5,('adv',75))],
 'DCA 5+5，逆向 100 點加':[(5,None),(5,('adv',100))],
 'DCA 4+3+3，逆向 40／80 加':[(4,None),(3,('adv',40)),(3,('adv',80))],
 'DCA 5+5，回到價值區邊緣（支撐）加':[(5,None),(5,('lvl','VA'))],
 'DCA 5+5，回到當天極值（強支撐）加':[(5,None),(5,('lvl','EXT'))],
 '只做 5 口、不加碼':[(5,None)],
 '順勢加碼 5+5，順向 25 點加':[(5,None),(5,('fav',25))],
 '順勢加碼 5+5，順向 40 點加':[(5,None),(5,('fav',40))],
}
rows=[]
for name,plan in PLANS.items():
    for sg in sigs:
        rr=run(sg,plan); d=g.dates[sg['i']]
        rows.append(dict(plan=name,date=d,net=rr['gross']-COST*rr['qty'],reason=rr['reason'],qty=rr['qty']))
T=pd.DataFrame(rows)
T['P']=np.select([T.date>='2026-10-01',T.date>='2026-01-01',T.date>='2025-01-01'],['4) 2026-10~11/10','3) 2026-01~09','2) 2025'],'1) 2024')
def dd(x): c=x.cumsum(); return (c-c.cummax()).min()
out=[]
for (pl,p),x in list(T.groupby(['plan','P'],sort=False))+[((pl,'全期'),x) for pl,x in T.groupby('plan',sort=False)]:
    out.append(dict(plan=pl,P=p,n=len(x),win=(x.net>0).mean()*100,total=x.net.sum(),avg=x.net.mean(),maxDD=dd(x.net),full_loss=(x.reason=='sl').sum(),added=(x.qty>x.qty.min()).mean()*100 if False else np.nan))
R=pd.DataFrame(out)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
order=list(PLANS)
for col,fmt in [('win',1),('total',0),('maxDD',0)]:
    t=R.pivot_table(index='plan',columns='P',values=col,sort=False).reindex(order).round(fmt); print('=====',col); print(t.to_string())
print(R[R.P=='全期'].set_index('plan').reindex(order)[['n','win','avg','full_loss']].round(1).to_string())
T.to_csv('dca_trades.csv',index=False)
