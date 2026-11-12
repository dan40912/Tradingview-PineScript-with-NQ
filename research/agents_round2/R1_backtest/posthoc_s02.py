"""事後檢查（不是選參數）：S02_top_off_B55、S02_ce_off_B55 的壓力與合併。研究用途，不是投資建議。"""
import pandas as pd, numpy as np, r1lib as R, run as RU
A=pd.read_pickle('all_combos.pkl'); SIGS=pd.read_pickle('all_sigs.pkl')
X=pd.read_pickle('analysis.pkl'); BT=X['BT']
def sh(st): return " / ".join(f"{st[k]['avg']:+,.0f}" if st[k]['n'] else "—" for k in R.SEGS)+f"（全期 {st['全期']['tot']:+,.0f} 回撤 {st['全期']['mdd']:+,.0f}）"
L=[]
for en in ['top','ce']:
    name=f'S02_{en}_off_B55'; v=A[('S02',name)]; T=v['T']; sp=v['sp']; sigs=SIGS[('S02',sp)]
    L.append(f"{name}：" + R.fmt_seg(R.seg_table(T)).replace("\n"," ｜"))
    Tc,_=RU.run_combo('S02',sigs,'B55',cost=96.0); L.append("  成本×2："+sh(R.seg_table(Tc)))
    f2=[f for s2,f in RU.sig_sets('S02',pen=2.0) if tuple(s2.items())==sp][0]; Ta,_=RU.run_combo('S02',f2(),'B55'); L.append("  限價穿 2 點才成交："+sh(R.seg_table(Ta)))
    Ts=T.copy(); Ts['pnl']-=80; L.append("  進出各滑 2 點："+sh(R.seg_table(Ts)))
    for nm,kw in [("停損緩衝+2",dict(stop_shift=2.0)),("停損緩衝-2",dict(stop_shift=-2.0))]:
        Tx,_=RU.run_combo('S02',sigs,'B55',**kw); L.append(f"  {nm}："+sh(R.seg_table(Tx)))
    dy=T.groupby('date').pnl.sum().sort_values(ascending=False); L.append(f"  拿掉最大 5 天：全期 {dy.iloc[5:].sum():+,.0f}（最大 5 天都是 +1,052 止盈）")
    nb=R.seg_table(T[~T.base_day]); L.append("  無基準日："+sh(nb))
    for sz in (1.0,0.5):
        M=RU.merged(T,size=sz); inc={s:M[M.seg==s].pnl.sum()-BT[s] for s in R.SEGS}
        L.append(f"  併入基準 {sz} 口：增量 "+" ".join(f"{s}{inc[s]:+,.0f}" for s in R.SEGS)+f"；合併全期 {M.pnl.sum():+,.0f} 回撤 {R.dd(M.sort_values(['date','e']).pnl.values):+,.0f}；擋掉基準 {520-int((M.src=='base').sum())} 筆")
    L.append(f"  同天同向基準 {100*T.base_same_day_dir.mean():.0f}%；測試期 t/p＝{RU.tstat(T[T.seg=='測試'].pnl.values)}")
open('posthoc_s02.txt','w').write("\n".join(L)); print("\n".join(L))
