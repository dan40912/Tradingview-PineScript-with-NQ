import pandas as pd, numpy as np, sys
NQ, ES = sys.argv[1], sys.argv[2]
def ld(p):
    t=pd.read_csv(p); t['et']=pd.to_datetime(t.entry_utc,utc=True); t['xt']=pd.to_datetime(t.exit_utc,utc=True)
    t['date']=(t.et.dt.tz_convert('America/New_York')+pd.Timedelta(hours=6)).dt.strftime('%Y-%m-%d'); return t
n=ld(NQ); e=ld(ES)
ROLL={'2024-03-12','2024-06-18','2024-09-17','2024-12-17','2025-03-18','2025-06-17','2025-09-16','2025-12-16','2026-03-17','2026-06-16','2026-09-15'}
n=n[~n.date.isin(ROLL)].copy()
ESK=['亞洲前低多','亞洲VAL多','早盤夜low多'.replace('low','低')]
c=n[n.kind.isin(ESK)].merge(e[e.kind.str.startswith('ES確認')].assign(kind=lambda x:x.kind.str[4:])[['date','kind','et']],on=['date','kind'],how='left',suffixes=('','_es'))
n=n.merge(c[['date','kind','et','et_es']],on=['date','kind','et'],how='left')
n['esok']=n.et_es.notna()&(n.et_es<=n.et)
def build(n, mode='順勢多點'):
    if mode=='順勢多點':
        cand=n[((n.tier==1)&(n.skipWhy.fillna('')!='逆盤中趨勢'))|(n.kind.isin(ESK)&n.esok)]
    else:  # 高勝率（只做多）：I2 多＋三個 ES 做多
        cand=n[((n.kind=='I2')&(n.dir==1))|(n.kind.isin(ESK)&n.esok)]
    cand=cand.sort_values('et'); taken=[]; why={}
    asia={}
    for idx,r in cand.iterrows():
        w=''
        if r.kind=='早盤夜低多' and not (r.vol>=1.2): w='波動不到1.2'
        elif r.kind in ('亞洲前低多','亞洲VAL多') and asia.get(r.date): w='今晚已有亞洲多'
        elif r.shock>=250: w='暴衝'
        else:
            day=[t for t in taken if t.date==r.date]
            real=sum(t.pnl for t in day if pd.notna(t.xt) and t.xt<r.et)   # exit in an earlier minute counts as realized
            op=sum(1 for t in day if not (pd.notna(t.xt) and t.xt<r.et))
            if real-2948*(op+1)<-3000: w='每日風險'
        if w: why[idx]=w; continue
        taken.append(r)
        if r.kind in ('亞洲前低多','亞洲VAL多'): asia[r.date]=True
    b=pd.DataFrame(taken); return b
if __name__=='__main__':
    b=build(n); b.to_csv('book6.csv',index=False)
    v=pd.read_csv('../final/V0_trades.csv')
    old=b[b.date<='2026-11-11']
    print('built (<=11/11):',len(old),'V0:',len(v),'win',f'{(old.pnl>0).mean():.1%}','total',old.pnl.sum())
    m=v.merge(old,left_on=['date','entry'],right_on=['date','entry'],how='outer',indicator=True)
    print(m._merge.value_counts().to_dict())
    print(m[m._merge!='both'][['date','key','kind','entry','pnl_x','pnl_y']].to_string())
