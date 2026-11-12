"""第一步：全部家族 × 止損 × 出場 粗掃。只輸出「探索」「驗證」兩段（測試／近期不看）。
輸出 explore_dv.csv。執行：python explore.py   （約數分鐘；需要 numba 會快很多）"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tlib
from families import FAMILIES, EXITS, STOPS, grid_iter, ctrl_long

D = None


def init():
    global D
    D = tlib.Data()


def st(T, P):
    x = T[T.P == P] if len(T) else T
    if len(x) == 0:
        return dict(n=0, win=np.nan, avg=np.nan, tot=0.0, mdd=0.0)
    p = x.pnl.values
    return dict(n=len(p), win=round(100 * (p > 0).mean(), 1), avg=round(p.mean(), 1), tot=round(p.sum()), mdd=round(tlib.dd(p)))


def job(args):
    fam, params = args
    if fam == "對照每天做多":
        sigs = ctrl_long(D, **params)
    else:
        sigs = FAMILIES[fam][0](D, **params)
    sigs = [s for s in sigs if D.P[s["i"]] in ("探索", "驗證")]   # 只看前兩段
    rows = []
    for sn, sc in STOPS.items():
        for xn, xc in EXITS.items():
            T = tlib.run(D, sigs, {**sc, **xc})
            r = dict(fam=fam, params=json.dumps(params, ensure_ascii=False), stop=sn, exit=xn, **params)
            for P, pre in (("探索", "d_"), ("驗證", "v_")):
                for k, v in st(T, P).items():
                    r[pre + k] = v
            rows.append(r)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(f, p) for f, (fn, G) in FAMILIES.items() for p in grid_iter(G)]
    jobs += [("對照每天做多", dict(N=N)) for N in (5, 15, 30)]
    print("進場組合", len(jobs), "× 止損", len(STOPS), "× 出場", len(EXITS), "=", len(jobs) * len(STOPS) * len(EXITS), "種變化")
    with Pool(4, initializer=init) as pool:
        res = pool.map(job, jobs, chunksize=4)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(tlib.OUT, "explore_dv.csv"), index=False)
    print("完成", len(df), "列", round(time.time() - t0), "秒")
