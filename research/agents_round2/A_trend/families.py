"""順勢進場家族（訊號產生器）。每個函式回傳 list[dict(i, e, d, stop_px)]：
e＝進場那根（RTH 索引），只用 e 之前（≤ e−1）已收盤的資料決定；stop_px＝結構止損價（可被出場設定改成固定點數）。
研究用途，不是投資建議。"""
from __future__ import annotations
import itertools
import numpy as np

SIDES = {"both": (1, -1), "long": (1,), "short": (-1,)}


def _days(D):
    return [i for i in range(D.n) if D.ok[i]]


# ---------- F1 開盤驅動 opening drive ----------
def f1_od(D, N=15, k=0.1, cl=0.0, es=0, side="both"):
    """09:30 起前 N 分鐘（第 0..N−1 根）淨移動 m = C[N−1] − O[0]；|m| ≥ k×ADR10 就順著做，
    cl>0：收盤要在前 N 分鐘區間的上（下）cl 比例內；es=1：ES 同期間同方向。第 N 根開盤進場。
    結構止損＝前 N 分鐘的另一端。"""
    out = []
    for i in _days(D):
        C, H, L, O = D.C[i], D.H[i], D.L[i], D.O[i]
        c = C[N - 1]
        if c != c:
            continue
        m = c - O[0]; hi = np.nanmax(H[:N]); lo = np.nanmin(L[:N])
        if abs(m) < k * D.adr10[i] or m == 0:
            continue
        d = 1 if m > 0 else -1
        if d not in SIDES[side]:
            continue
        if cl > 0:
            pos = (c - lo) / max(hi - lo, 1e-9)
            if (d == 1 and pos < 1 - cl) or (d == -1 and pos > cl):
                continue
        if es:
            em = D.eC[i, N - 1] - D.eO[i, 0]
            if not (em == em) or np.sign(em) != d:
                continue
        out.append(dict(i=i, e=N, d=d, stop_px=lo if d == 1 else hi))
    return out


F1_GRID = dict(N=[5, 10, 15, 30], k=[0.0, 0.05, 0.1, 0.15, 0.2], cl=[0.0, 0.3], es=[0, 1], side=["both", "long"])


# ---------- F2 開盤區間突破 ORB ----------
def f2_orb(D, N=15, cut=150, filt="none", side="both"):
    """開盤前 N 分鐘高低＝區間；之後到 cut（RTH 索引）前，第一根 1 分 K 收盤突破區間就順著做（每天最多一筆），
    下一根開盤進場。filt：none／oov（09:30 開在前日價值區外）／es（ES 同一根也收在它自己的區間外同方向）／
    narrow（區間寬 < 前 10 天同區間平均）。結構止損＝區間另一側。"""
    out = []
    oravg = D.or_avg(N)
    for i in _days(D):
        C, H, L, O = D.C[i], D.H[i], D.L[i], D.O[i]
        hi = np.nanmax(H[:N]); lo = np.nanmin(L[:N])
        if filt == "oov" and (D.val[i] <= O[0] <= D.vah[i]):
            continue
        if filt == "narrow" and not (hi - lo < oravg[i]):
            continue
        if filt == "es":
            ehi = np.nanmax(D.eH[i, :N]); elo = np.nanmin(D.eL[i, :N])
        for t in range(N, cut):
            c = C[t]
            if c != c:
                continue
            d = 1 if c > hi else (-1 if c < lo else 0)
            if d == 0:
                continue
            if filt == "es":
                ec = D.eC[i, t]
                if not ((d == 1 and ec > ehi) or (d == -1 and ec < elo)):
                    continue
            if d in SIDES[side]:
                out.append(dict(i=i, e=t + 1, d=d, stop_px=lo if d == 1 else hi))
            break
    return out


F2_GRID = dict(N=[5, 15, 30], cut=[90, 150], filt=["none", "oov", "es", "narrow"], side=["both", "long"])


# ---------- F3 VWAP 趨勢 ----------
def f3_vwap(D, start=30, K=30, slope=0.0, mode="pull", last=150, side="both"):
    """t ≥ start：最近 K 根收盤全在 VWAP 同一側，且 VWAP 比 30 分鐘前同向移動 > slope×ADR10。
    mode=pull：這根最低（做空看最高）碰到 VWAP+0.02×ADR 以內、收盤仍在 VWAP 同一側 → 下一根進場；
    mode=first：條件第一次成立就下一根進場。每天每個方向最多一筆，last 之後不進場。
    結構止損＝訊號根的 VWAP 再退 0.05×ADR10。"""
    out = []
    for i in _days(D):
        C, H, L, VW = D.C[i], D.H[i], D.L[i], D.vwap[i]
        a = D.adr10[i]
        above = C > VW; below = C < VW
        done = set()
        for t in range(max(start, K, 30), last):
            if C[t] != C[t]:
                continue
            for d in SIDES[side]:
                if d in done:
                    continue
                side_ok = above[t - K + 1:t + 1].all() if d == 1 else below[t - K + 1:t + 1].all()
                if not side_ok:
                    continue
                sl = VW[t] - VW[t - 30]
                if not (d * sl > slope * a):
                    continue
                if mode == "pull":
                    touch = (L[t] <= VW[t] + 0.02 * a) if d == 1 else (H[t] >= VW[t] - 0.02 * a)
                    if not touch:
                        continue
                out.append(dict(i=i, e=t + 1, d=d, stop_px=VW[t] - d * 0.05 * a))
                done.add(d)
    return out


F3_GRID = dict(start=[30, 60], K=[15, 30, 60], slope=[0.0, 0.03], mode=["pull", "first"], last=[150, 270], side=["both", "long"])


# ---------- F4 盤中創新高／新低動能 ----------
def f4_newhi(D, T=60, mv=0.0, vw=1, last=240, side="both"):
    """t ≥ T：這根收盤高於當天（09:30 起、到上一根為止）的最高價 → 做多（低於最低價 → 做空），
    mv：當時價格離 09:30 開盤至少 mv×ADR10（同方向）；vw=1：收盤在 VWAP 同一側。每天最多一筆。
    結構止損＝VWAP 再退 0.05×ADR10。"""
    out = []
    for i in _days(D):
        C, H, L, O, VW = D.C[i], D.H[i], D.L[i], D.O[i], D.vwap[i]
        a = D.adr10[i]
        hh = np.fmax.accumulate(np.where(np.isnan(H), -np.inf, H))
        ll = np.fmin.accumulate(np.where(np.isnan(L), np.inf, L))
        for t in range(T, last):
            c = C[t]
            if c != c:
                continue
            d = 1 if c > hh[t - 1] else (-1 if c < ll[t - 1] else 0)
            if d == 0:
                continue
            if d * (c - O[0]) < mv * a:
                continue
            if vw and d * (c - VW[t]) <= 0:
                continue
            if d in SIDES[side]:
                out.append(dict(i=i, e=t + 1, d=d, stop_px=VW[t] - d * 0.05 * a))
            break
    return out


F4_GRID = dict(T=[30, 60, 90], mv=[0.0, 0.1, 0.2], vw=[0, 1], last=[180, 300], side=["both", "long"])


# ---------- F5a 夜盤方向＋開盤確認 ----------
def f5_on(D, N=15, k=0.1, vw=1, side="both"):
    """夜盤淨變動 on = 09:29 價 − 前一天 RTH 收盤；|on| ≥ k×ADR10。到 09:30+N 時，C[N−1] 相對 09:30 開盤同方向，
    且（vw=1）在 VWAP 同一側 → 第 N 根開盤順著做。結構止損＝開盤後 N 分鐘另一端。"""
    out = []
    for i in _days(D):
        on = D.on_last[i] - D.pdc[i]
        if not (on == on) or abs(on) < k * D.adr10[i] or on == 0:
            continue
        d = 1 if on > 0 else -1
        if d not in SIDES[side]:
            continue
        c = D.C[i, N - 1]
        if c != c or d * (c - D.O[i, 0]) <= 0:
            continue
        if vw and d * (c - D.vwap[i, N - 1]) <= 0:
            continue
        hi = np.nanmax(D.H[i, :N]); lo = np.nanmin(D.L[i, :N])
        out.append(dict(i=i, e=N, d=d, stop_px=lo if d == 1 else hi))
    return out


F5A_GRID = dict(N=[5, 15, 30], k=[0.0, 0.1, 0.2, 0.3], vw=[0, 1], side=["both", "long"])


# ---------- F5b NQ 相對 ES 強弱 ----------
def f5_rs(D, N=30, th=0.0, need_es=1, side="both"):
    """到 09:30+N：rN＝NQ 相對 09:30 報酬、rE＝ES 報酬。NQ 方向 d＝sign(rN)；
    要求 d×(rN − rE) > th（NQ 比 ES 更往該方向走，單位 %）；need_es=1：ES 也同方向。
    第 N 根開盤進場；結構止損＝開盤後 N 分鐘另一端。"""
    out = []
    for i in _days(D):
        c, e = D.C[i, N - 1], D.eC[i, N - 1]
        if c != c or e != e:
            continue
        rN = (c / D.O[i, 0] - 1) * 100; rE = (e / D.eO[i, 0] - 1) * 100
        if rN == 0:
            continue
        d = 1 if rN > 0 else -1
        if d not in SIDES[side]:
            continue
        if need_es and np.sign(rE) != d:
            continue
        if d * (rN - rE) <= th:
            continue
        hi = np.nanmax(D.H[i, :N]); lo = np.nanmin(D.L[i, :N])
        out.append(dict(i=i, e=N, d=d, stop_px=lo if d == 1 else hi))
    return out


F5B_GRID = dict(N=[15, 30, 60], th=[0.0, 0.05, 0.1], need_es=[0, 1], side=["both", "long"])


# ---------- F6a 開盤試探失敗後收復開盤價（open reclaim） ----------
def f6_reclaim(D, T=30, k=0.1, vw=1, last=150, side="both"):
    """t ≥ T：當天已經往反方向走過 ≥ k×ADR10（例如先跌到開盤−k×ADR），之後第一根收盤回到 09:30 開盤價之上
    （且 vw=1 時在 VWAP 之上）→ 下一根做多；鏡像做空。每天最多一筆。結構止損＝當天到目前的最低（做空最高）。
    （這是 11/11 那種「先跌、收復開盤、一路漲」的型態）"""
    out = []
    for i in _days(D):
        C, H, L, O, VW = D.C[i], D.H[i], D.L[i], D.O[i], D.vwap[i]
        a = D.adr10[i]; o = O[0]
        ll = np.fmin.accumulate(np.where(np.isnan(L), np.inf, L))
        hh = np.fmax.accumulate(np.where(np.isnan(H), -np.inf, H))
        # 開盤後一開始就在上方的不算「收復」：要求上一根收盤在開盤另一側
        for t in range(T, last):
            c = C[t]
            if c != c:
                continue
            pc = C[t - 1]
            d = 0
            if c > o and pc <= o and o - ll[t] >= k * a:
                d = 1
            elif c < o and pc >= o and hh[t] - o >= k * a:
                d = -1
            if d == 0:
                continue
            if vw and d * (c - VW[t]) <= 0:
                continue
            if d in SIDES[side]:
                out.append(dict(i=i, e=t + 1, d=d, stop_px=ll[t] if d == 1 else hh[t]))
                break
    return out


F6A_GRID = dict(T=[15, 30, 45], k=[0.05, 0.1, 0.15, 0.2], vw=[0, 1], last=[120, 180], side=["both", "long"])


# ---------- F6b 趨勢日（上午單邊）→ 中午後順勢抱到收盤 ----------
def f6_trendday(D, at=120, k=0.2, frac=0.9, side="both"):
    """在 09:30+at 時：從 10:00 到現在的收盤有 ≥ frac 比例在 VWAP 同一側，且離 09:30 開盤 ≥ k×ADR10 → 下一根順著做。
    結構止損＝VWAP 再退 0.05×ADR10。"""
    out = []
    for i in _days(D):
        C, VW, O = D.C[i], D.vwap[i], D.O[i]
        c = C[at - 1]
        if c != c:
            continue
        m = c - O[0]; a = D.adr10[i]
        if abs(m) < k * a:
            continue
        d = 1 if m > 0 else -1
        if d not in SIDES[side]:
            continue
        seg = C[30:at] - VW[30:at]; seg = seg[np.isfinite(seg)]
        if len(seg) == 0 or (np.sign(seg) == d).mean() < frac:
            continue
        out.append(dict(i=i, e=at, d=d, stop_px=VW[at - 1] - d * 0.05 * a))
    return out


F6B_GRID = dict(at=[90, 120, 180, 240], k=[0.1, 0.2, 0.3], frac=[0.8, 0.95], side=["both", "long"])


# ---------- F6c 跳空延續 gap-and-go ----------
def f6_gap(D, N=15, gk=0.1, hold=1, side="both"):
    """gap＝09:30 開盤 − 前一天 RTH 收盤，|gap| ≥ gk×ADR10；開盤後 N 分鐘沒有回補一半跳空（hold=1：還要收在開盤價同側）→
    第 N 根順跳空方向做。結構止損＝前 N 分鐘另一端。"""
    out = []
    for i in _days(D):
        gp = D.O[i, 0] - D.pdc[i]
        if not (gp == gp) or abs(gp) < gk * D.adr10[i] or gp == 0:
            continue
        d = 1 if gp > 0 else -1
        if d not in SIDES[side]:
            continue
        hi = np.nanmax(D.H[i, :N]); lo = np.nanmin(D.L[i, :N])
        half = D.pdc[i] + gp / 2
        if (d == 1 and lo <= half) or (d == -1 and hi >= half):
            continue
        c = D.C[i, N - 1]
        if c != c or (hold and d * (c - D.O[i, 0]) <= 0):
            continue
        out.append(dict(i=i, e=N, d=d, stop_px=lo if d == 1 else hi))
    return out


F6C_GRID = dict(N=[5, 15, 30], gk=[0.05, 0.1, 0.2, 0.3], hold=[0, 1], side=["both", "long"])


# ---------- 對照組：每天同一時間無條件做多 ----------
def ctrl_long(D, N=15):
    return [dict(i=i, e=N, d=1, stop_px=np.nanmin(D.L[i, :N])) for i in _days(D)]


FAMILIES = {
    "F1開盤驅動": (f1_od, F1_GRID),
    "F2開盤區間突破": (f2_orb, F2_GRID),
    "F3VWAP趨勢": (f3_vwap, F3_GRID),
    "F4盤中新高低": (f4_newhi, F4_GRID),
    "F5a夜盤方向": (f5_on, F5A_GRID),
    "F5b相對ES": (f5_rs, F5B_GRID),
    "F6a收復開盤": (f6_reclaim, F6A_GRID),
    "F6b趨勢日": (f6_trendday, F6B_GRID),
    "F6c跳空延續": (f6_gap, F6C_GRID),
}


def grid_iter(G):
    keys = list(G)
    for vals in itertools.product(*[G[k] for k in keys]):
        yield dict(zip(keys, vals))


# 出場設定（stop 由 STOPS 決定）
EXITS = {
    "收盤": dict(),
    "移40": dict(trail=40, ttrig="same"),
    "移70": dict(trail=70, ttrig="same"),
    "移100": dict(trail=100, ttrig="same"),
    "移150": dict(trail=150, ttrig="same"),
    "移A.15": dict(trail=0.15, trail_adr=True, ttrig="same"),
    "移A.25": dict(trail=0.25, trail_adr=True, ttrig="same"),
    "VWAP": dict(vwbuf=0.0),
    "VWAP+移100": dict(vwbuf=0.0, trail=100, ttrig="same"),
    "2R": dict(tp_R=2.0),
    "3R": dict(tp_R=3.0),
    "保本1R": dict(be_R=1.0),
    "15根低": dict(tbars=15, ttrig_R=1.0),
    "30根低": dict(tbars=30, ttrig_R=1.0),
    "中午": dict(xidx=179),
}
STOPS = {
    "結構": dict(stop_mode="px"),
    "100點": dict(stop_mode="pts", stop_pts=100.0),
    "0.25ADR": dict(stop_mode="adr", stop_adr=0.25),
}
