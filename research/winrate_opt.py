"""勝率最佳化：在每段期望值都為正的前提下，找勝率最高的做法。
加入 day-trader-skill 的方法：成交量確認（trader-entry --min-volume-x）、ATR 停損（trader-exit atr:K）、
滾動歷史回放（trader-timemachine 的 history），以及 VWAP 位置、ORB15 方向。
勝率＝扣成本後損益 > 0 的比例。開發 2024–2025、驗證 2026-01～09、選完才看 2026-10。"""
import pickle, itertools, numpy as np, pandas as pd, grid
from multiprocessing import Pool
from eng import sim
nan = np.isnan
nq = pickle.load(open("NQ_days2.pkl", "rb")); ESd = grid.ES
# 每天 5 分K 成交量，以及「過去 20 天同時段平均」→ 相對成交量 RVOL
V5 = np.array([np.nansum(d["V"].reshape(78, 5), 1) for d in nq])
AVG = np.full_like(V5, np.nan)
for k in range(20, len(nq)): AVG[k] = V5[k - 20:k].mean(0)
for k, d in enumerate(nq): d["k"] = k
def rvol(d, i): return V5[d["k"], i // 5] / AVG[d["k"], i // 5] if not nan(AVG[d["k"], 0]) else np.nan
def atr5(d, i, n=14):
    H, L, C = d["H"][:i + 1].reshape(-1)[: (i + 1) // 5 * 5].reshape(-1, 5), d["L"][: (i + 1) // 5 * 5].reshape(-1, 5), d["C"][: (i + 1) // 5 * 5].reshape(-1, 5)
    h, l = np.nanmax(H, 1), np.nanmin(L, 1); c = np.array([r[~nan(r)][-1] if (~nan(r)).any() else np.nan for r in C])
    pc = np.r_[np.nan, c[:-1]]; tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], 1); tr = tr[~nan(tr)]
    return tr[-n:].mean() if len(tr) else np.nan
def orb_dir(d, upto):
    orh, orl = np.nanmax(d["H"][:15]), np.nanmin(d["L"][:15])
    for i in range(19, upto + 1, 5):
        c = d["C"][i]
        if nan(c): continue
        if c > orh: return 1
        if c < orl: return -1
    return 0
def filters_ok(d, i, side, vw, orbf, vol):
    if vw and np.sign(d["C"][i] - d["vwap"][i]) != side: return False
    if orbf and orb_dir(d, i) == -side: return False
    if vol and not (rvol(d, i) >= vol): return False
    return True

def i2x(d, last=89, h1=False, tp=55, sl=145, xi=179, es=False, vw=False, orbf=False, vol=0, be=None):
    O, C, val, vah = d["O"], d["C"], d["val"], d["vah"]
    if nan(O[0]): return None
    o = O[0]; s = -1 if o > vah else 1 if o < val else 0
    if s == 0 or (h1 and ((o - vah) if s == -1 else (val - o)) > vah - val) or (es and ESd.get(d["date"], 0) != s): return None
    cnt = 0
    for i in range(4, 120, 5):
        if nan(C[i]): continue
        cnt = cnt + 1 if val <= C[i] <= vah else 0
        if cnt >= 2:
            ei = i + 1
            if 28 <= ei <= 32 or ei > last or not filters_ok(d, i, s, vw, orbf, vol): return None
            return sim(d, ei, s, tp, sl, xi, be, None, "I2")
    return None

def trendx(d, rr=2.0, stop="ib", xi=179, vw=False, vol=0, be=1.0):
    O, H, L, C, val, vah = d["O"], d["H"], d["L"], d["C"], d["val"], d["vah"]
    if nan(O[0]): return None
    o = O[0]; g = 1 if o > vah else -1 if o < val else 0
    if g == 0 or not (o > d["pdh"] or o < d["pdl"]): return None
    ibh, ibl = np.nanmax(H[:30]), np.nanmin(L[:30]); mid = (ibh + ibl) / 2
    for i in range(4, 150, 5):
        if nan(C[i]): continue
        if val <= C[i] <= vah: return None
        if i >= 34 and ((C[i] > ibh) if g == 1 else (C[i] < ibl)):
            ei = i + 1
            if ei > 389 or nan(O[ei]) or not filters_ok(d, i, g, vw, False, vol): return None
            s = min(max(abs(O[ei] - mid), 30), 80) if stop == "ib" else min(max(float(stop[3:]) * atr5(d, i), 15), 120)
            return sim(d, ei, g, rr * s, s, xi, be, None, "順勢突破")
    return None

def refx(d, which=1, tp=55, sl=145, xi=179, vw=False, vol=0, be=None):
    H, L, C = d["H"], d["L"], d["C"]; hi = lo = None
    for i in range(0, 120):
        if nan(H[i]): continue
        hi = H[i] if hi is None else max(hi, H[i]); lo = L[i] if lo is None else min(lo, L[i])
        if i >= 4 and i % 5 == 4 and 5 <= i + 1 <= 120 and not 28 <= i + 1 <= 32:
            s = -1 if which == 0 else 1
            if (which == 0 and hi > d["pdh"] and C[i] < d["pdh"]) or (which == 1 and lo < d["pdl"] and C[i] > d["pdl"]):
                if not filters_ok(d, i, s, vw, False, vol): return None
                return sim(d, i + 1, s, tp, sl, xi, be, None, "前高空" if which == 0 else "前低多")
    return None

def variants():
    V = []
    for last, h1, es, vw, orbf, vol, (tp, sl), xi, be in itertools.product((59, 89), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1.0), ((30, 145), (40, 145), (55, 145), (30, 100), (40, 100), (55, 100)), (179, 379), (None, 0.5)):
        V.append(("I2", f"I2 最晚{last} h1={h1} es={es} vwap={vw} orb={orbf} rvol≥{vol} +{tp}/-{sl} 出場{xi} 保本={be}",
                  dict(last=last, h1=bool(h1), es=bool(es), vw=bool(vw), orbf=bool(orbf), vol=vol, tp=tp, sl=sl, xi=xi, be=be)))
    for rr, stop, xi, vw, vol, be in itertools.product((0.5, 1.0, 1.5, 2.0), ("ib", "atr1.5", "atr2.5"), (179, 379), (0, 1), (0, 1.0, 1.5), (None, 1.0, 0.5)):
        V.append(("順勢突破", f"順勢 {rr}R 停損{stop} 出場{xi} vwap={vw} rvol≥{vol} 保本={be}", dict(rr=rr, stop=stop, xi=xi, vw=bool(vw), vol=vol, be=be)))
    for w, vw, vol, (tp, sl), be in itertools.product((0, 1), (0, 1), (0, 1.0), ((30, 145), (55, 145), (30, 100), (55, 100)), (None, 0.5)):
        V.append(("參考點", f"{'前高空' if w == 0 else '前低多'} vwap={vw} rvol≥{vol} +{tp}/-{sl} 保本={be}", dict(which=w, vw=bool(vw), vol=vol, tp=tp, sl=sl, be=be)))
    return V
FN = {"I2": i2x, "順勢突破": trendx, "參考點": refx}
P = {"2024": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31"), "2026驗證": ("2026-01-01", "2026-09-30"), "10月": ("2026-10-01", "2026-10-31"), "選擇期": ("2024-01-01", "2026-09-30")}
def run(v):
    fam, name, kw = v; f = FN[fam]
    tr = [t | {"date": d["date"]} for d in nq for t in [f(d, **kw)] if t]
    df = pd.DataFrame(tr) if tr else pd.DataFrame(columns=["date", "pnl", "res"])
    r = dict(family=fam, name=name)
    for lab, (a, b) in P.items():
        x = df[(df.date >= a) & (df.date <= b)].pnl.values if len(df) else np.array([])
        r[f"{lab}筆"] = len(x); r[f"{lab}勝率"] = (x > 0).mean() if len(x) else np.nan; r[f"{lab}每筆"] = x.mean() if len(x) else np.nan; r[f"{lab}合計"] = x.sum()
        if lab == "選擇期" and len(x): c = np.cumsum(x); r["選擇期回撤"] = (np.maximum.accumulate(np.r_[0, c])[1:] - c).max()
    return r
if __name__ == "__main__":
    V = variants(); print("假設數", len(V))
    with Pool(4) as p: R = pd.DataFrame(p.map(run, V, chunksize=16))
    R.to_pickle("winrate_grid.pkl"); print("完成")
