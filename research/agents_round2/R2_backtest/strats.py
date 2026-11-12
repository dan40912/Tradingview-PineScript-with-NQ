"""五個修正版策略（Q2_critic/strategies_v2.json）的訊號＋模擬。研究用途，不是投資建議。
每個 gen_*() 回傳『未經每日 $3,000 檢查』的逐筆交易（DataFrame），並在每列記錄 last_data_slot（訊號用到的最後一根 K 的 slot），
以及檢查用的數值（chk_* 欄）。exec 參數：entry='open'|'adverse'、slip（點）、cost、late（晚幾分鐘進場）、size。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import r2lib as R
from r2lib import N, O, H, L, C, V, EC, slot, sim, price_at, entry_slot_valid

EX = dict(entry="open", slip=0.0, cost=R.COST, late=0, size=1.0)


def _ex(kw):
    e = dict(EX); e.update(kw); return e


def _sim(i, e, d, sl, xs, ex, cond=None):
    return sim(i, e + ex["late"], d, sl, xs, cond_exit=cond, entry=ex["entry"], slip=ex["slip"], cost=ex["cost"], size=ex["size"])


# =====================================================================  Q03v2  Noise Area
CK = [slot(f"{h:02d}:{m:02d}") for h in range(10, 16) for m in (0, 30) if (h, m) <= (15, 30)]   # 10:00..15:30（12 個）
_q03 = None


def _q03_pre():
    global _q03
    if _q03 is not None:
        return _q03
    nT = len(CK)
    P = np.full((N, nT), np.nan); PS = np.full((N, nT), -1)
    VW = np.full((N, nT), np.nan)
    O930 = np.full(N, np.nan)
    tp = (H + L + C) / 3
    for i in range(N):
        k = entry_slot_valid(i, R.R0, 3)
        O930[i] = O[i, k] if k >= 0 else np.nan
        m = np.isfinite(tp[i]) & np.isfinite(V[i]) & (V[i] > 0)
        pv = np.cumsum(np.where(m, tp[i] * V[i], 0.0)); vv = np.cumsum(np.where(m, V[i], 0.0))
        for t, s in enumerate(CK):
            P[i, t], PS[i, t] = price_at(C, i, s)
            a = pv[s - 1] - pv[R.R0 - 1]; b = vv[s - 1] - vv[R.R0 - 1]
            VW[i, t] = a / b if b > 0 else np.nan
    MOVE = np.abs(P / O930[:, None] - 1)
    SIG = np.full((N, nT), np.nan)
    for t in range(nT):
        SIG[:, t] = R.prior_mean(MOVE[:, t], 14, R.g.tradable, 0.75)
    _q03 = dict(P=P, PS=PS, VW=VW, O930=O930, SIG=SIG, MOVE=MOVE)
    return _q03


def gen_q03(m=1.0, trail="vwap_band", start="10:00", **kw):
    ex = _ex(kw); Z = _q03_pre()
    P, PS, VW, O930, SIG = Z["P"], Z["PS"], Z["VW"], Z["O930"], Z["SIG"]
    st = slot(start)
    rows = []
    for i in range(N):
        if not (R.TRAD[i] and R.PREV_SAME[i] and np.isfinite(O930[i]) and np.isfinite(R.ADR10[i]) and np.isfinite(R.PDC[i])):
            continue
        up = max(O930[i], R.PDC[i]); lo = min(O930[i], R.PDC[i])
        UB = up * (1 + m * SIG[i]); LB = lo * (1 - m * SIG[i])
        sl = min(145.0, 0.25 * R.ADR10[i])
        ntr = 0; next_ok = st
        for t, s in enumerate(CK):
            if ntr >= 2:
                break
            if s < st or s < next_ok:
                continue
            p = P[i, t]
            if not (np.isfinite(p) and np.isfinite(SIG[i, t])):
                continue
            d = 1 if p > UB[t] else (-1 if p < LB[t] else 0)
            if d == 0:
                continue
            cond = set(); last_used = PS[i, t]
            if trail == "vwap_band":
                for t2 in range(t + 1, len(CK)):
                    p2 = P[i, t2]
                    if not np.isfinite(p2) or not np.isfinite(SIG[i, t2]):
                        continue
                    vw = VW[i, t2]
                    if d == 1:
                        lvl = UB[t2] if not np.isfinite(vw) else max(UB[t2], vw)
                        if p2 < lvl: cond.add(CK[t2])
                    else:
                        lvl = LB[t2] if not np.isfinite(vw) else min(LB[t2], vw)
                        if p2 > lvl: cond.add(CK[t2])
            r = _sim(i, s, d, sl, R.EOD, ex, cond)
            if r is None:
                continue
            r.update(T=R.hhmm(s), last_data_slot=int(max(last_used, s - 1)), chk_p=p, chk_p_slot=R.hhmm(PS[i, t]),
                     chk_O930=O930[i], chk_PDC=R.PDC[i], chk_sigma=SIG[i, t], chk_UB=UB[t], chk_LB=LB[t],
                     chk_vwap=VW[i, t], chk_ADR10=R.ADR10[i], ntr=ntr + 1)
            rows.append(r)
            ntr += 1
            nxt = [c for c in CK if c > r["x"]]
            if not nxt or r["x"] >= R.EOD:
                break
            next_ok = nxt[0]
    return pd.DataFrame(rows)


# =====================================================================  Q13v2  NQ vs ES 相對強弱
_q13 = {}


def _beta60():
    if "beta" in _q13:
        return _q13["beta"]
    rn = R.RTH_CLOSE / R.RTH_OPEN - 1; re = R.ES_CLOSE / R.ES_OPEN - 1
    beta = np.full(N, np.nan); hist = []
    for i in range(N):
        if len(hist) >= 60:
            w = np.array(hist[-60:]); ok = np.isfinite(w).all(1)
            if ok.sum() >= 45:
                x, y = w[ok, 1], w[ok, 0]
                beta[i] = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
        if R.g.tradable[i]:
            hist.append((rn[i], re[i]))
    _q13["beta"] = beta
    return beta


def _rel(T):
    if T in _q13:
        return _q13[T]
    s = slot(T); beta = _beta60()
    rel = np.full(N, np.nan); used = np.full(N, -1)
    for i in range(N):
        a, sa = price_at(C, i, s); b, sb = price_at(C, i, s - 60)
        ea, esa = price_at(EC, i, s); eb, esb = price_at(EC, i, s - 60)
        if np.isfinite([a, b, ea, eb, beta[i]]).all():
            rel[i] = np.log(a / b) - beta[i] * np.log(ea / eb)
            used[i] = max(sa, esa)
    # z：前 20 個真正 RTH 日的 rel 標準差（不含當天，≥15 個有效）
    sd = np.full(N, np.nan); hist = []
    for i in range(N):
        if len(hist) >= 20:
            w = np.array(hist[-20:]); w = w[np.isfinite(w)]
            if len(w) >= 15:
                sd[i] = np.std(w, ddof=1)
        if R.g.tradable[i]:
            hist.append(rel[i])
    _q13[T] = (rel, sd, used)
    return _q13[T]


def gen_q13(decision_time="09:45", z0=1.0, nostop=False, **kw):
    ex = _ex(kw); s = slot(decision_time)
    rel, sd, used = _rel(decision_time); beta = _beta60()
    xs = s + 119
    rows = []
    for i in range(N):
        if not (R.TRAD[i] and np.isfinite(R.ADR10[i]) and np.isfinite(rel[i]) and np.isfinite(sd[i]) and sd[i] > 0):
            continue
        z = rel[i] / sd[i]
        if z0 == 0:
            d = 1 if z > 0 else (-1 if z < 0 else 0)
        else:
            d = 1 if z >= z0 else (-1 if z <= -z0 else 0)
        if d == 0:
            continue
        sl = 1e6 if nostop else min(145.0, 0.35 * R.ADR10[i])
        r = _sim(i, s, d, sl, xs, ex)
        if r is None:
            continue
        if nostop:
            r["worst"] = 0.0
        r.update(last_data_slot=int(used[i]), chk_beta=beta[i], chk_rel_bp=rel[i] * 1e4, chk_sd_bp=sd[i] * 1e4, chk_z=z,
                 chk_ADR10=R.ADR10[i], chk_nq_now=price_at(C, i, s)[0], chk_nq_60=price_at(C, i, s - 60)[0])
        rows.append(r)
    return pd.DataFrame(rows)


# =====================================================================  Q02v2  大位移日 15:30 順勢
_q02 = {}


def _sigd20():
    if "s" in _q02:
        return _q02["s"]
    ret = R.RTH_CLOSE / R.PDC - 1          # PDC 只在同合約才有值
    s = np.full(N, np.nan); hist = []
    for i in range(N):
        if len(hist) >= 20:
            s[i] = np.std(hist[-20:], ddof=1)
        if R.g.tradable[i] and np.isfinite(ret[i]):
            hist.append(ret[i])
    _q02["s"] = s
    return s


def gen_q02(k=1.0, SL_pts=120, reverse=False, **kw):
    ex = _ex(kw); sg = _sigd20(); s = slot("15:30")
    rows = []
    for i in range(N):
        if not (R.TRAD[i] and R.PREV_SAME[i] and np.isfinite(R.PDC[i]) and np.isfinite(sg[i])):
            continue
        p, ps = price_at(C, i, s)
        if not np.isfinite(p):
            continue
        rod = p / R.PDC[i] - 1
        if abs(rod) < k * sg[i] or rod == 0:
            continue
        d = int(np.sign(rod)) * (-1 if reverse else 1)
        r = _sim(i, s, d, float(SL_pts), R.EOD, ex)
        if r is None:
            continue
        r.update(last_data_slot=int(ps), chk_p1529=p, chk_PDC=R.PDC[i], chk_ROD_pct=rod * 100, chk_sigd20_pct=sg[i] * 100,
                 chk_ratio=abs(rod) / sg[i])
        rows.append(r)
    return pd.DataFrame(rows)


# =====================================================================  Q08v2  CPI∪NFP 公布前夜盤做多
def gen_q08(entry_time="03:00", SL_pts=80, days="event", **kw):
    """days='event'：CPI∪NFP；'nonevent'：其他所有有資料的日子（對照）。"""
    ex = _ex(kw); s = slot(entry_time); xs = slot("08:28")
    rows = []
    for i in range(N):
        if R.SEG[i] == "無":
            continue
        isev = R.DATES[i] in R.DATA_DAYS
        if (days == "event") != isev:
            continue
        r = _sim(i, s, 1, float(SL_pts), xs, ex)
        if r is None:
            continue
        r.update(last_data_slot=s - 1, chk_event=("CPI" if R.DATES[i] in R.CPI else "") + ("NFP" if R.DATES[i] in R.NFP else ""))
        rows.append(r)
    return pd.DataFrame(rows)


# =====================================================================  EVT_FOMC
def gen_evt(rule="A", days="event", q=0.15, **kw):
    ex = _ex(kw); rows = []
    for i in range(N):
        if not R.TRAD[i]:
            continue
        isev = R.DATES[i] in R.FOMC
        if (days == "event") != isev:
            continue
        if rule == "A":
            r = _sim(i, slot("09:31"), 1, 120.0, slot("13:55"), ex)
            if r is None:
                continue
            r.update(last_data_slot=slot("09:31") - 1)
        else:
            if not np.isfinite(R.ADR10[i]):
                continue
            a, sa = price_at(C, i, slot("15:00")); b, sb = price_at(C, i, slot("14:00"))
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            dl = a - b
            if abs(dl) < q * R.ADR10[i] or dl == 0:
                continue
            d = -int(np.sign(dl))
            r = _sim(i, slot("15:00"), d, 100.0, R.EOD, ex)
            if r is None:
                continue
            r.update(last_data_slot=int(sa), chk_c1359=b, chk_c1459=a, chk_delta=dl, chk_thr=q * R.ADR10[i])
        rows.append(r)
    return pd.DataFrame(rows)
