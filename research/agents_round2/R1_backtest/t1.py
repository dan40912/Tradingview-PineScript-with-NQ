import time, strat as S, r1lib as R
i11=R.g.dindex['2026-11-11']
for nm,f,args in [("S11",S.gen_S11,(5,False)),("S11es",S.gen_S11,(5,True)),("S06",S.gen_S06,("NQ","AM")),("S03",S.gen_S03,("mid",)),
                  ("S02",S.gen_S02,("ote","2R",False)),("S10",S.gen_S10,("now","off")),("S01",S.gen_S01,("AM","any")),
                  ("S07",S.gen_S07,("off",)),("S09",S.gen_S09,("AV","limit")),("S09r",S.gen_S09,("AV","reject"))]:
    t=time.time(); sg=f(*args); 
    x=[s for s in sg if s['i']==i11]
    print(nm, len(sg), f"{time.time()-t:.1f}s", [(s['d'],R.hhmm(s.get('e',s.get('e0'))),s['stop_raw'],s['info']) for s in x])
print(S.SKIP)
