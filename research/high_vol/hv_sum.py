import pandas as pd, numpy as np
r=pd.read_csv("hv_trades.csv")
r["P"]=np.where(r.td<"2026-01-01","P1",np.where(r.td<"2026-10-01","P2","P3"))
def tab(x):
    g=x.groupby(["strat","param","exit","P"]).pnl.agg(n="size",win=lambda s:(s>0).mean(),avg="mean").unstack("P")
    return g
hi=tab(r[r.vm>=1.8]); mid=tab(r[r.vm>=1.5]); lo=tab(r[r.vm<1.5])
hi["通過"]=[(row[("n","P1")]>=15 and row[("n","P2")]>=15 and row[("avg","P1")]>0 and row[("avg","P2")]>0) for _,row in hi.iterrows()]
pd.set_option("display.width",250)
cols=[("n","P1"),("win","P1"),("avg","P1"),("n","P2"),("win","P2"),("avg","P2"),("n","P3"),("win","P3"),("avg","P3")]
print("##### 波動 ≥1.8 倍"); print(hi[cols+[("通過","")]].round(2).to_string())
print("\n##### 波動 ≥1.5 倍（每筆平均）"); print(mid[[("n","P1"),("avg","P1"),("n","P2"),("avg","P2"),("n","P3"),("avg","P3")]].round(0).to_string())
print("\n##### 對照：波動 <1.5 倍（每筆平均）"); print(lo[[("n","P1"),("avg","P1"),("n","P2"),("avg","P2")]].round(0).to_string())
for s in ["S1","S2","S3","S4"]:
    x=hi.loc[s]; print(s,"通過",int(x["通過"].sum()),"/",len(x))
print("\n11/25:"); print(r[r.td=="2026-11-25"][["strat","param","exit","side","entry","et","risk","pnl"]].round(0).to_string(index=False))
