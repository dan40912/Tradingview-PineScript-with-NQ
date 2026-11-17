# 事先登記：進場改成「訊號後掛限價單，比原進場價再便宜 k 點才買（空單貴 k 點才賣）」，TP/SL 從成交價算 +55/−145，
# 有效時間 30 分鐘或到原本的出場時間；成交那分鐘不准止盈；同分鐘兩邊都碰到算止損；沒成交＝不做。
# 通過條件：探索、驗證、測試三段「合計」和「勝率」都比現行（下一根開盤市價進場）好。近期只報告。
import pandas as pd, numpy as np
A=pd.read_csv('A_book_mae.csv',parse_dates=['et','xt'])
d=pd.read_csv('../wt/data/NQ_1m_continuous_20240102-20261117.csv.gz',usecols=['timestamp_utc','open','high','low','close'])
ts=pd.to_datetime(d.timestamp_utc,utc=True); et=ts.dt.tz_convert('America/New_York'); M=(et.dt.hour*60+et.dt.minute).values
O,H,L,C=[d[c].values for c in ['open','high','low','close']]
idx={t:i for i,t in enumerate(ts.values.astype('datetime64[m]').astype(np.int64))}
ASIA={'亞洲前低多','亞洲VAL多'}
def deadline(i,asia):
    j=i
    while j+1<len(M):
        m=M[j]
        if asia and 179<=m<1080: return j
        if not asia and (m>=749 and m<=959): return j
        if not asia and (m<570 or m>959): return j-1
        j+=1
    return j
def sim(r,k,win):
    i=idx[np.datetime64(r.et.tz_convert(None),'m').astype(np.int64)]; asia=r.kind in ASIA; D=deadline(i,asia); s=r.dir
    lim=r.entry - s*k; last=D if win is None else min(D,i+win)
    f=None
    for j in range(i,last+1):
        if (s==1 and L[j]<=lim) or (s==-1 and H[j]>=lim):
            px=min(lim,O[j]) if s==1 else max(lim,O[j]); f=j; break
    if f is None: return None
    tp=px+s*55; sl=px-s*145
    for j in range(f,D+1):
        if (s==1 and L[j]<=sl) or (s==-1 and H[j]>=sl): return -145*20-48
        if j>f and ((s==1 and H[j]>=tp) or (s==-1 and L[j]<=tp)): return 55*20-48
    return s*(C[D]-px)*20-48
P=['1探索','2驗證','3測試','4近期10/1起']
res=[]
base=A.groupby('P').pnl.agg(['size','sum',lambda x:(x>0).mean()]); base.columns=['n','sum','wr']
res.append(dict(variant='現行（市價）',**{f'{p}':f"{int(base.loc[p,'n'])}筆 {base.loc[p,'wr']:.0%} {base.loc[p,'sum']:+,.0f}" for p in P},通過='基準'))
for win in [30,None]:
    for kk in ['10','20','30','50','75','0.1A','0.2A']:
        out=[]
        for r in A.itertuples():
            k=float(kk) if 'A' not in kk else float(kk[:-1])*r.vol*255.2
            out.append(sim(r,k,win))
        A['v']=out; g={}
        ok=True
        for p in P:
            x=A[A.P==p]; f=x.v.dropna()
            g[p]=f"{len(f)}筆 {(f>0).mean():.0%} {f.sum():+,.0f}"
            if p!='4近期10/1起' and not (f.sum()>base.loc[p,'sum'] and (f>0).mean()>base.loc[p,'wr']): ok=False
        res.append(dict(variant=f'便宜 {kk} 點／有效 {"30分" if win else "到出場"}',**g,通過='通過' if ok else '沒通過'))
pd.set_option('display.width',250); out=pd.DataFrame(res); print(out.to_string(index=False)); out.to_csv('entry_test_result.csv',index=False)
