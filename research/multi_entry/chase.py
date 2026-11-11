exec(open('gapgo.py').read().split('rows=[]')[0])
sp=lambda d: 'disc' if d<='2025-06-30' else 'val' if d<='2025-12-31' else 'test' if d<='2026-10-19' else 'fwd'
out=[]
for th in [100,150,200,250,300]:
  for side in (1,-1):
    for k in range(1,len(days)):
        a,b=G[days[k-1]],G[days[k]]; ra=a[a.is_rth==1]
        if len(ra)<300 or a.contract.iloc[-1]!=b.contract.iloc[0]: continue
        pc=ra.close.iloc[-1]; pre=b[b.sm<930]
        if len(pre)<500: continue
        hit=pre[((pre.high>=pc+th) if side==1 else (pre.low<=pc-th))&(pre.sm<=840)]
        if len(hit): out.append(dict(th=th,side=side,date=days[k],p=sim(b,hit.index[0]+1,side,869)*20-48,sm=hit.sm.iloc[0]))
O=pd.DataFrame(out); O['sp']=O.date.map(sp)
t=O.pivot_table(index=['side','th'],columns='sp',values='p',aggfunc='mean').round(0)[['disc','val','test','fwd']]
n=O.pivot_table(index=['side','th'],columns='sp',values='p',aggfunc='size')[['disc','val','test','fwd']]
w=O.groupby(['side','th']).p.agg(lambda x:(x>0).mean()*100).round(1)
print(pd.concat([t.add_suffix('_avg'),n.add_suffix('_n'),w.rename('win%')],axis=1).to_string())
O.to_csv('chase.csv',index=False)
