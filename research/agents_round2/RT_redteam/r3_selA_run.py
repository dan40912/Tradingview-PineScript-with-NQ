"""檢查 3（選擇偏誤，A）：用 A 自己的家族程式，把 29,475 種變化的「測試／近期」也跑出來（A 只存了探索／驗證）。
這支程式刻意 import A 的程式——目的是量 A 的家族在測試期的基準率，不是獨立重算。輸出 rt_selA_all.csv。"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
A_DIR = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/agents/A_trend"
sys.path.insert(0, A_DIR)
import tlib
from families import FAMILIES, EXITS, STOPS, grid_iter, ctrl_long
D = None
def init():
    global D
    D = tlib.Data()
def job(args):
    fam, params = args
    sigs = ctrl_long(D, **params) if fam == "對照每天做多" else FAMILIES[fam][0](D, **params)
    rows = []
    for sn, sc in STOPS.items():
        for xn, xc in EXITS.items():
            T = tlib.run(D, sigs, {**sc, **xc})
            r = dict(fam=fam, params=json.dumps(params, ensure_ascii=False), stop=sn, exit=xn)
            for P, pre in (("探索", "d_"), ("驗證", "v_"), ("測試", "t_"), ("近期", "r_")):
                x = T[T.P == P] if len(T) else T
                r[pre + "n"] = len(x); r[pre + "avg"] = x.pnl.mean() if len(x) else np.nan; r[pre + "tot"] = x.pnl.sum() if len(x) else 0.0
            rows.append(r)
    return rows
if __name__ == "__main__":
    t0 = time.time()
    jobs = [(f, p) for f, (fn, G) in FAMILIES.items() for p in grid_iter(G)] + [("對照每天做多", dict(N=N)) for N in (5, 15, 30)]
    with Pool(4, initializer=init) as pool:
        res = pool.map(job, jobs, chunksize=4)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rt_selA_all.csv"), index=False)
    print("完成", len(df), round(time.time() - t0), "秒")
