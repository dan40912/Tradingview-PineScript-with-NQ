"""亞洲／倫敦／美盤的盤別進場點（2026-11-06 對話中測的 7 種規則）。
用法：python3 session_rules.py [起始交易日]   例：python3 session_rules.py 2026-10-20
沒給日期 = 全期間（約 2 分鐘），印出每條規則×止盈止損的每年每筆平均；給日期 = 只列 A1 在那天之後的逐筆。
研究用途，不是投資建議。"""
import sys, os, glob
import pandas as pd, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fs=sorted(glob.glob(os.path.join(ROOT,'data','NQ_1m_continuous_*.csv.gz')))+sorted(glob.glob(os.path.join(ROOT,'data','NQ_1m_yahoo_*.csv.gz')))
df=pd.concat([pd.read_csv(f,usecols=['timestamp_utc','open','high','low','close','volume','contract','trade_date','is_rth','day_quality']) for f in fs],ignore_index=True)
START=sys.argv[1] if len(sys.argv)>1 else None
if START: df=df[df.trade_date>=str(pd.Timestamp(START)-pd.Timedelta(days=7))[:10]]
df['t']=pd.to_datetime(df.timestamp_utc,utc=True).dt.tz_convert('America/New_York')
df['hm']=df.t.dt.hour*60+df.t.dt.minute
df=df[df.day_quality!='exclude']
df['sm']=np.where(df.hm>=1080,df.hm-1080,df.hm+360)  # minutes since 18:00 ET
COST=48; PT=20
ASIA=(0,539)      # 18:00-02:59 ET
LON=(540,869)     # 03:00-08:29 ET
RTH0=930          # 09:30
days=sorted(df.trade_date.unique())
g={d:x.sort_values('t').reset_index(drop=True) for d,x in df.groupby('trade_date')}
# prior RTH hi/lo, contract check
prev={}
for i in range(1,len(days)):
    a=g[days[i-1]]; r=a[a.is_rth==1]
    if len(r)>300 and a.contract.iloc[-1]==g[days[i]].contract.iloc[0]:
        prev[days[i]]=(r.high.max(),r.low.min(),r.close.iloc[-1])

def sim(x,i0,dirn,tp,sl,tend):
    e=x.open[i0]; T=e+dirn*tp; S=e-dirn*sl
    for i in range(i0,len(x)):
        if x.sm[i]>tend: return (x.close[i-1]-e)*dirn,'time'
        h,l=x.high[i],x.low[i]
        hitS = l<=S if dirn>0 else h>=S
        hitT = h>=T if dirn>0 else l<=T
        if hitS: return -sl,'sl'
        if hitT: return tp,'tp'
    return (x.close.iloc[-1]-e)*dirn,'time'

def fives(x,a,b):
    w=x[(x.sm>=a)&(x.sm<=b)]
    k=(w.sm//5)
    agg=w.groupby(k).agg(close=('close','last'),high=('high','max'),low=('low','min'),last=('sm','max'))
    return agg

def signals(x,d):
    out=[]
    asia=x[(x.sm>=ASIA[0])&(x.sm<=ASIA[1])]
    lon=x[(x.sm>=LON[0])&(x.sm<=LON[1])]
    on=x[x.sm<RTH0]
    if len(asia)<400 or len(lon)<250: return out
    AH,AL=asia.high.max(),asia.low.min()
    LH,LL=lon.high.max(),lon.low.min()
    OH,OL=on.high.max(),on.low.min()
    def fade(name,lvl_hi,lvl_lo,a,b,last_entry,tend):
        f=fives(x,a,b)
        sw_h=sw_l=False; done_s=done_l=False
        for k,r in f.iterrows():
            if r['last']>last_entry: break
            if r.high>lvl_hi: sw_h=True
            if r.low<lvl_lo: sw_l=True
            ent=np.searchsorted(x.sm.values,r['last']+1)
            if ent>=len(x): break
            if sw_h and not done_s and r.close<lvl_hi:
                out.append((name,-1,ent,tend)); done_s=True
            if sw_l and not done_l and r.close>lvl_lo:
                out.append((name,1,ent,tend)); done_l=True
    def brk(name,lvl_hi,lvl_lo,a,b,last_entry,tend):
        f=fives(x,a,b)
        for k,r in f.iterrows():
            if r['last']>last_entry: break
            ent=np.searchsorted(x.sm.values,r['last']+1)
            if r.close>lvl_hi: out.append((name,1,ent,tend)); break
            if r.close<lvl_lo: out.append((name,-1,ent,tend)); break
    if d in prev:
        ph,pl,pc=prev[d]
        fade('A1 亞洲盤：掃前日RTH高低後收回(反向)',ph,pl,0,539,480,539)
        brk('A2 亞洲盤：突破前日RTH高低(順勢)',ph,pl,0,539,480,539)
    fade('L1 倫敦盤：掃亞洲高低後收回(反向)',AH,AL,540,869,810,869)
    brk('L2 倫敦盤：突破亞洲高低(順勢)',AH,AL,540,869,810,869)
    fade('N1 美盤：掃隔夜高低後收回(反向)',OH,OL,930,1050,1050,1109)
    brk('N2 美盤：突破隔夜高低(順勢)',OH,OL,934,1050,1050,1109)
    fade('N3 美盤：掃倫敦高低後收回(反向)',LH,LL,930,1050,1050,1109)
    return out


if __name__=="__main__":
    rows=[]
    GRID=[(55,145)] if START else [(20,20),(30,60),(55,55),(55,145)]
    for d in days[1:]:
        if START and d<START: continue
        x=g[d]
        for name,dirn,ent,tend in signals(x,d):
            if START and not name.startswith('A1'): continue
            for tp,sl in GRID:
                p,why=sim(x,ent,dirn,tp,sl,tend)
                rows.append(dict(rule=name,date=d,yr=d[:4],dir=dirn,entry_et=x.t[ent].strftime('%Y-%m-%d %H:%M'),
                                 entry=x.open[ent],tp=tp,sl=sl,why=why,net=p*PT-COST))
    r=pd.DataFrame(rows)
    pd.set_option('display.width',250)
    if START: print(r.to_string(index=False)); print('n',len(r),'total',r.net.sum(),'avg',round(r.net.mean(),0))
    else: print(r.groupby(['rule','tp','sl','yr']).net.mean().unstack().round(0))
