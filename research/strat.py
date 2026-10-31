"""所有假設。每個函式回傳 0 或 1 筆交易（dict）；TP／SL 都寫在參數裡。"""
import numpy as np
from eng import sim, FOMC
nan = np.isnan

def i2(D, last=120, h1=False, tp=55, sl=145, xi=179, ha=None, be=None, inside_pd=False, es=None, vol=None):
    O, C, val, vah = D["O"], D["C"], D["val"], D["vah"]
    if nan(O[0]): return None
    o = O[0]; d = -1 if o > vah else 1 if o < val else 0
    if d == 0: return None
    dist = (o - vah) if d == -1 else (val - o)
    if h1 and dist > vah - val: return None
    if inside_pd and (o > D["pdh"] or o < D["pdl"]): return None
    if es is not None and es.get(D["date"], 0) != d: return None
    if vol and not (vol[0] <= D["pr_rel"] <= vol[1]): return None
    cnt = 0
    for i in range(4, 120, 5):
        if nan(C[i]): continue
        cnt = cnt + 1 if val <= C[i] <= vah else 0
        if cnt >= 2:
            ei = i + 1
            if 28 <= ei <= 32 or ei > last: return None
            return sim(D, ei, d, tp, sl, xi, be, ha, "I2")
    return None

def trend(D, ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="any", be=None, last=150, es=None, vol=None, fomc=True):
    O, H, L, C, val, vah = D["O"], D["H"], D["L"], D["C"], D["val"], D["vah"]
    if nan(O[0]) or (not fomc and D["date"] in FOMC): return None
    o = O[0]; g = 1 if o > vah else -1 if o < val else 0
    if g == 0: return None
    dist = (o - vah) if g == 1 else (val - o); outpd = o > D["pdh"] or o < D["pdl"]; far = dist > vah - val
    if not {"any": outpd or far, "outpd": outpd, "all": True}[cond]: return None
    if es is not None and es.get(D["date"], 0) != -g: return None     # es 字典存的是 I2 方向（-1＝開在 VAH 上）
    if vol and not (vol[0] <= D["pr_rel"] <= vol[1]): return None
    ibh, ibl = np.nanmax(H[:ib + 1]), np.nanmin(L[:ib + 1]); mid = (ibh + ibl) / 2
    for i in range(4, last, 5):
        if nan(C[i]): continue
        c = C[i]
        if val <= c <= vah: return None
        if i >= ib + 5 and ((c > ibh) if g == 1 else (c < ibl)):
            ei = i + 1
            if ei > 389 or nan(O[ei]): return None
            s = min(max(abs(O[ei] - mid), smin), smax)
            return sim(D, ei, g, rr * s, s, xi, be, None, "順勢突破")
    return None

def pullback(D, ib=29, rr=2.0, smin=20, smax=60, xi=179, last=210):
    """順勢回踩：順勢突破成立後，5 分K 回測 IB 邊緣（±5 點）又收回 → 進場；停損在回測低點外 5 點。"""
    O, H, L, C, val, vah = D["O"], D["H"], D["L"], D["C"], D["val"], D["vah"]
    if nan(O[0]): return None
    o = O[0]; g = 1 if o > vah else -1 if o < val else 0
    if g == 0: return None
    dist = (o - vah) if g == 1 else (val - o)
    if not (o > D["pdh"] or o < D["pdl"] or dist > vah - val): return None
    ibh, ibl = np.nanmax(H[:ib + 1]), np.nanmin(L[:ib + 1]); lvl = ibh if g == 1 else ibl; broke = False
    for i in range(4, last, 5):
        if nan(C[i]): continue
        c = C[i]
        if val <= c <= vah: return None
        if not broke:
            if i >= ib + 5 and ((c > ibh) if g == 1 else (c < ibl)): broke = True
            continue
        wl, wh = np.nanmin(L[i - 4:i + 1]), np.nanmax(H[i - 4:i + 1])
        if (wl <= lvl + 5 and c > lvl) if g == 1 else (wh >= lvl - 5 and c < lvl):
            ei = i + 1
            if ei > 389 or nan(O[ei]): return None
            ext = (wl - 5) if g == 1 else (wh + 5); s = min(max(abs(O[ei] - ext), smin), smax)
            return sim(D, ei, g, rr * s, s, xi, None, None, "順勢回踩")
    return None

def orb(D, orl=15, rr=2.0, smin=20, smax=80, xi=179, filt="none", last=90):
    """開盤區間突破：前 orl 分鐘的高低點，之後 5 分K 收盤突破 → 進場；停損在區間中點。"""
    O, H, L, C = D["O"], D["H"], D["L"], D["C"]
    if nan(O[0]): return None
    orh, orlo = np.nanmax(H[:orl]), np.nanmin(L[:orl]); mid = (orh + orlo) / 2; o = O[0]
    for i in range(orl + 4, last, 5):
        if nan(C[i]): continue
        c = C[i]; sd = 1 if c > orh else -1 if c < orlo else 0
        if sd == 0: continue
        if filt == "gap" and sd != (1 if o > D["pclose"] else -1): return None
        if filt == "vwap" and (c - D["vwap"][i]) * sd <= 0: return None
        if filt == "va" and not ((sd == 1 and o > D["vah"]) or (sd == -1 and o < D["val"])): return None
        ei = i + 1
        if 28 <= ei <= 32 or ei > 389 or nan(O[ei]): return None
        s = min(max(abs(O[ei] - mid), smin), smax)
        return sim(D, ei, sd, rr * s, s, xi, None, None, f"ORB{orl}")
    return None

def ibfade(D, ext=5, rr=1.5, smin=15, smax=60, xi=179, last=180):
    """IB 假突破反轉：10:00 後突破 IB 高（低）點至少 ext 點，5 分K 又收回 IB 內 → 反向。"""
    O, H, L, C = D["O"], D["H"], D["L"], D["C"]
    if nan(O[0]): return None
    ibh, ibl = np.nanmax(H[:30]), np.nanmin(L[:30]); hi = lo = None
    for i in range(34, last, 5):
        if nan(C[i]): continue
        hi = np.nanmax(H[30:i + 1]); lo = np.nanmin(L[30:i + 1]); c = C[i]
        sd = -1 if (hi >= ibh + ext and c < ibh) else 1 if (lo <= ibl - ext and c > ibl) else 0
        if sd:
            ei = i + 1
            if ei > 389 or nan(O[ei]): return None
            ext_px = hi + 5 if sd == -1 else lo - 5; s = min(max(abs(O[ei] - ext_px), smin), smax)
            return sim(D, ei, sd, rr * s, s, xi, None, None, "IB假突破")
    return None

def vwap_pb(D, rr=2.0, smin=20, smax=60, xi=379, last=240):
    """VWAP 順勢回踩：10:00–10:30 每根 5 分K 都收在 VWAP 同一邊 → 之後第一次碰到 VWAP 又收回 → 順勢進場。"""
    O, H, L, C, vw = D["O"], D["H"], D["L"], D["C"], D["vwap"]
    if nan(O[0]): return None
    cl = [C[i] - vw[i] for i in range(34, 60, 5) if not nan(C[i])]
    if len(cl) < 4: return None
    g = 1 if all(x > 0 for x in cl) else -1 if all(x < 0 for x in cl) else 0
    if g == 0: return None
    for i in range(64, last, 5):
        if nan(C[i]): continue
        wl, wh = np.nanmin(L[i - 4:i + 1]), np.nanmax(H[i - 4:i + 1])
        if (wl <= vw[i] and C[i] > vw[i]) if g == 1 else (wh >= vw[i] and C[i] < vw[i]):
            ei = i + 1
            if ei > 389 or nan(O[ei]): return None
            ext = wl - 5 if g == 1 else wh + 5; s = min(max(abs(O[ei] - ext), smin), smax)
            return sim(D, ei, g, rr * s, s, xi, None, None, "VWAP回踩")
    return None

def pdfade(D, which=1, tp=55, sl=145, xi=179):
    """原本的參考點：前低假突破多（which=1）、前高假突破空（which=0）。"""
    O, H, L, C = D["O"], D["H"], D["L"], D["C"]; hi = lo = None
    for i in range(0, 120):
        if nan(H[i]): continue
        hi = H[i] if hi is None else max(hi, H[i]); lo = L[i] if lo is None else min(lo, L[i])
        if i >= 4 and i % 5 == 4:
            ei = i + 1
            if not (5 <= ei <= 120) or 28 <= ei <= 32: continue
            if which == 0 and hi > D["pdh"] and C[i] < D["pdh"]: return sim(D, ei, -1, tp, sl, xi, None, None, "前高空")
            if which == 1 and lo < D["pdl"] and C[i] > D["pdl"]: return sim(D, ei, 1, tp, sl, xi, None, None, "前低多")
    return None
