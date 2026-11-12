"""交叉驗證：r2lib.sim 與 B_exits core._sim（numba，599/599 驗證過）在『只有停損、時間出場』時逐筆比對。"""
import sys; sys.dont_write_bytecode=True
import numpy as np, r2lib as R, strats as S, b_core as B
tot=0; bad=0
for name,T in [("Q13v2",S.gen_q13("09:45",0.0)),("Q13v2_1300",S.gen_q13("13:00",0.0)),("Q02v2",S.gen_q02(0.5,60)),("Q08v2",S.gen_q08("03:00",120)),
               ("EVT_A",S.gen_evt("A")),("Q03 hard_only",S.gen_q03(1.0,"hard_only","10:00"))]:
    nb=0
    for r in T.itertuples():
        xs = {"Q08v2":868,"EVT_A":1195}.get(name, R.EOD if name in ("Q02v2","Q03 hard_only") else r.e+119)
        if name=="Q13v2_1300": xs=1259
        if name=="Q13v2": xs=1064
        p=B.plan(sl=r.sl,tp1=np.inf)
        i=r.i
        real,filled,xsl,why=B._sim(R.O[i],R.H[i],R.L[i],R.C[i],int(r.e),int(xs),int(r.d),p)
        pnl=real*20-48*filled
        if abs(pnl-r.pnl)>0.01 or xsl!=r.x:
            nb+=1
            if nb<=3: print(name,r.date,r.e,r.pnl,pnl,r.x,xsl)
    print(f"{name}: {len(T)} 筆，不一致 {nb}"); tot+=len(T); bad+=nb
print("合計",tot,"不一致",bad)
