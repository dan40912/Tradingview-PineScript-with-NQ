import pandas as pd, numpy as np, r1lib as R
X=pd.read_pickle('analysis.pkl'); DET=X['DET']; BT=X['BT']; GR=X['GR']; SEL=X['SEL']
L=[]
for sid,d in DET.items():
    T=d['T']; st=R.seg_table(T); a=st['全期']
    nb=d['nobase']; nbt=sum(nb[s]['tot'] for s in R.SEGS)
    m1=d['merged']['基準＋本策略 1 口']; m5=d['merged']['基準＋本策略 0.5 口']
    inc1={s:m1[m1.seg==s].pnl.sum()-BT[s] for s in R.SEGS}; inc5={s:m5[m5.seg==s].pnl.sum()-BT[s] for s in R.SEGS}
    L.append(f"{sid} {SEL[sid]}｜每筆 探/驗/測/近 " + "/".join(f"{st[s]['avg']:+.0f}" if st[s]['n'] else '—' for s in R.SEGS)
      + f"（{'/'.join(str(st[s]['n']) for s in R.SEGS)} 筆）｜全期 {a['tot']:+,.0f} 回撤 {a['mdd']:+,.0f}"
      + f"｜無基準日 {nb['全期']['n']}筆 {nbt:+,.0f}（探/驗/測 " + "/".join(f"{nb[s]['tot']:+,.0f}" for s in R.SEGS[:3]) + "）"
      + f"｜併入基準 1口 測試增量 {inc1['測試']:+,.0f} 全期 {sum(inc1.values()):+,.0f}；0.5口 測試 {inc5['測試']:+,.0f} 全期 {sum(inc5.values()):+,.0f}｜{GR[sid][0]}")
open('table.txt','w').write("\n".join(L)); print("\n".join(L))
