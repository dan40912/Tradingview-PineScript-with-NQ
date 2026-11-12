"""2026-11-11 當天各規則的動作，以及 C1 參數的實際點數。"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tlib
from candidates import CANDIDATES, cfg
from families import f1_od, f6_gap, STOPS, EXITS
from lib.sim import hhmm_of
D = tlib.Data()
i = list(D.dates).index("2026-11-11")
a = D.adr10[i]
print(f"11/11：ADR10={a:.1f}，0.15×ADR={0.15*a:.1f} 點，0.25×ADR={0.25*a:.1f} 點；09:30 開 {D.O[i,0]}；前一天收 {D.pdc[i]}（跳空 {D.O[i,0]-D.pdc[i]:+.2f}）")
for N in (5, 10, 15, 30):
    print(f"  前 {N} 分鐘：收 {D.C[i,N-1]}，淨移動 {D.C[i,N-1]-D.O[i,0]:+.2f}（= {(D.C[i,N-1]-D.O[i,0])/a:+.3f} ADR）")
for N in (5, 10, 15):
    for k in (0.1, 0.15, 0.2):
        T = tlib.run(D, [s for s in f1_od(D, N=N, k=k) if s["i"] == i], {**STOPS["0.25ADR"], **EXITS["移100"]})
        for r in T.itertuples():
            print(f"  開盤驅動 N={N} k={k}：{'多' if r.d==1 else '空'} {hhmm_of(r.e+930)} {r.entry} → {hhmm_of(r.x+930)} {r.exit}（{r.why}）{r.pnl:+,.0f}")
for nm in CANDIDATES:
    T = pd.read_csv(os.path.join(tlib.OUT, f"trades_{nm}.csv"))
    x = T[T.date == "2026-11-11"]
    print(nm, "無訊號" if len(x) == 0 else x[["d", "e", "x", "entry", "exit", "stop0", "why", "pnl"]].to_dict("records"))
# ADR 與止損點數分佈
ok = D.ok
for P in tlib.PERIODS:
    m = ok & (D.P == P)
    s = np.minimum(0.25 * D.adr10[m], 145)
    print(f"{P}：ADR10 中位 {np.median(D.adr10[m]):.0f}；0.25ADR 止損中位 {np.median(s):.0f} 點，被 145 封頂的比例 {(0.25*D.adr10[m] > 145).mean()*100:.0f}%；0.15ADR 門檻中位 {np.median(0.15*D.adr10[m]):.0f} 點")
T = pd.read_csv(os.path.join(tlib.OUT, "trades_C1_開盤驅動5分.csv"))
print("C1 止損點數中位", T.groupby("P").dist.median().round(0).to_dict(), "；持有分鐘中位", (T.x - T.e).median())
print("C1 最近 10 筆："); print(T.tail(10)[["date", "d", "entry", "exit", "dist", "why", "pnl"]].to_string(index=False))
