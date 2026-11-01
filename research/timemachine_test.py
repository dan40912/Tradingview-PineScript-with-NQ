"""借用 day-trader-skill 的三個想法，在 NQ 上測：
1) Time Machine：用前 60 天同時段的 5 分K 隨機重組剩下的走勢（虛無假設＝沒有方向性的隨機路徑），
   比較實際損益和隨機路徑的期望值；再測「只做 Time Machine 期望值 > 0 的單」。
2) 固定金額風險的口數（trader-budget）：用 MNQ，每筆風險固定。
3) 它的預設風控（reward:risk ≥ 1、不能放空）套在我們的策略上會怎樣。"""
import sys, pickle, numpy as np, pandas as pd
import grid, strat
nq = pickle.load(open("NQ_days2.pkl", "rb")); idx = {d["date"]: k for k, d in enumerate(nq)}
def bars5(d):
    O, H, L, C = (d[k].reshape(78, 5) for k in ("O", "H", "L", "C"))
    with np.errstate(all="ignore"):
        o = O[:, 0]; h = np.nanmax(H, 1); l = np.nanmin(L, 1); c = np.array([r[~np.isnan(r)][-1] if (~np.isnan(r)).any() else np.nan for r in C])
    return np.c_[o, h, l, c]
B5 = [bars5(d) for d in nq]
full = [k for k, b in enumerate(B5) if not np.isnan(b).any()]
def null_ev(k, t, n=2000, window=2, seed=19):
    """trader-timemachine 的算法：每根 5 分K 從前 60 個完整交易日、同時段 ±2 根裡隨機抽一根，用比例接到現價。"""
    pool = [j for j in full if j < k][-60:]
    if len(pool) < 20: return np.nan, np.nan, np.nan
    P = np.stack([B5[j] for j in pool])
    prevc = np.concatenate([np.full((len(pool), 1, 1), np.nan), P[:, :-1, 3:4]], 1)
    R = P / prevc                                     # (天, 78, 4) 對前一根收盤的比例
    rng = np.random.default_rng(seed); b0, b1 = t["ei"] // 5, min(t["xi_cap"] // 5, 77)
    e = nq[k]["O"][t["ei"]]; side = t["side"]; sl = t["risk"]; tp = t["tp"]
    price = np.full(n, e); done = np.zeros(n, bool); pnl = np.zeros(n)
    for b in range(b0, b1 + 1):
        j = np.clip(b + rng.integers(-window, window + 1, n), 1, 77); dd = rng.integers(0, len(pool), n)
        r = R[dd, j]; o, h, l, c = (price * r[:, q] for q in range(4))
        if b == b0: h = np.maximum(h, e); l = np.minimum(l, e)
        hit_sl = ~done & ((l <= e - sl) if side == 1 else (h >= e + sl)); pnl[hit_sl] = -sl; done |= hit_sl
        hit_tp = ~done & ((h >= e + tp) if side == 1 else (l <= e - tp)); pnl[hit_tp] = tp; done |= hit_tp
        price = c
    pnl[~done] = side * (price[~done] - e)
    return pnl.mean() * 20 - 48, (pnl == tp).mean(), (pnl == -sl).mean()
imp_es = dict(last=89, h1=False, tp=55, sl=145, xi=379, es=grid.ES)
T1 = dict(ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="outpd", be=None)   # Time Machine 不模擬保本停損，這裡先關掉
rows = []
for name, f, cap in (("I2(ES同向)", lambda d: strat.i2(d, **imp_es), 379), ("順勢突破30", lambda d: strat.trend(d, **T1), 179)):
    for d in nq:
        t = f(d)
        if not t: continue
        k = idx[d["date"]]; tp = 55 if name.startswith("I2") else 2 * t["risk"]
        ev, ptp, psl = null_ev(k, t | {"xi_cap": cap, "tp": tp})
        rows.append(dict(strategy=name, date=d["date"], side=t["side"], pnl=t["pnl"], null_ev=ev, null_tp=ptp, null_sl=psl, risk=t["risk"]))
X = pd.DataFrame(rows).dropna(); X.to_pickle("borrow_tm.pkl")
print("— 1) Time Machine 虛無模型：實際 vs 隨機路徑 —")
for s, g in X.groupby("strategy"):
    print(f"{s:<10} {len(g)} 筆｜實際每筆 {g.pnl.mean():+5.0f}｜隨機路徑期望值 {g.null_ev.mean():+5.0f}｜差（真正來自走勢結構的優勢）{g.pnl.mean()-g.null_ev.mean():+5.0f}"
          f"｜隨機路徑先碰 TP {g.null_tp.mean():.0%}、先碰 SL {g.null_sl.mean():.0%}")
    for lab, m in (("Time Machine 期望值 > 0 的單", g.null_ev > 0), ("Time Machine 期望值 ≤ 0 的單", g.null_ev <= 0)):
        x = g[m]; print(f"     {lab:<24} {len(x):3d} 筆 每筆實際 {x.pnl.mean():+5.0f}")
