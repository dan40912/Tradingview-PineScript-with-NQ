"""補充：基準＋「趨勢日同向的 T1（VWAP 回檔）」（不加 TD 單）。研究用途，不是投資建議。"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import a4_alloc as AL, a3_trend as A, cbook as B
for q in [0.5, 0.67, 0.8]:
    for k in [2, 3, 4]:
        dv = A.td_signal("10", q, k)
        x = AL.tsel(dv, rules=("T1",))
        T = B.book(extra=x); b = B.brief(T)
        print(f"q{q} k{k} T1同向 {len(x):3d}筆 每筆 " + " ".join(f"{p}:{x[x.P==p].pnl.mean():+5.0f}" for p in B.PER) +
              " ｜組合 " + "｜".join(f"{p} {b[p][0]:+,.0f}/{b[p][1]:+,.0f}" for p in B.PER + ["全期"]))
