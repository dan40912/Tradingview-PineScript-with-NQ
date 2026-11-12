"""R1_backtest：8 個策略的訊號產生器（研究用途，不是投資建議）。
每個產生器回傳每天最多 1 個訊號 dict：
  i, d, kind('mkt'|'lim'), e0(市價原進場 slot)／e(限價成交 slot), P(限價價位), stop_raw(結構停損價，QC2 前),
  tgt(結構目標價或 None＝2R), no_tp(EOD 版), info(進場前用到的數值與時間), known(用到的資料最晚 slot)
規則照 Q1 strategies_v2.json；含糊處的保守解讀寫在每個函式的說明，報告會整理。
"""
from __future__ import annotations
import numpy as np
from r1lib import *  # noqa: F401,F403
import r1lib as R

# =====================================================================================
# 共用：每日 RTH 高低（給 N 日極值）
with np.errstate(all="ignore"):
    RH_NQ = np.nanmax(R.H[:, R0:R0 + 390], 1); RL_NQ = np.nanmin(R.L[:, R0:R0 + 390], 1)
    RH_ES = np.nanmax(R.EH[:, R0:R0 + 390], 1); RL_ES = np.nanmin(R.EL[:, R0:R0 + 390], 1)
TRADABLE_ALL = R.g.tradable


def nday_level(i, n, es=False):
    """前 n 個 RTH 交易日（g.tradable）的最低低點／最高高點，以及極值所在日距今的交易日數（同值取最近）。
    任何一天和當天不同合約 → None（QC7）。"""
    con = R.ECONTRACT if es else R.CONTRACT
    rh, rl = (RH_ES, RL_ES) if es else (RH_NQ, RL_NQ)
    js = []
    j = i - 1
    while j >= 0 and len(js) < n:
        if TRADABLE_ALL[j]:
            js.append(j)
        j -= 1
    if len(js) < n or any(con[j] != con[i] for j in js):
        return None
    lows = np.array([rl[j] for j in js]); highs = np.array([rh[j] for j in js])   # js[0]＝昨天（lag 1）
    if not (np.isfinite(lows).all() and np.isfinite(highs).all()):
        return None
    lo = lows.min(); hi = highs.max()
    lag_lo = int(np.where(lows == lo)[0].min()) + 1
    lag_hi = int(np.where(highs == hi)[0].min()) + 1
    return dict(lo=lo, hi=hi, lag_lo=lag_lo, lag_hi=lag_hi, lo_day=R.DATES[js[lag_lo - 1]], hi_day=R.DATES[js[lag_hi - 1]])


def soup_event(Lr, Hr, Cr, d, lvl, brk, rec, end_slot=1079):   # 11:59 那分鐘收盤＝12:00
    """09:30 後第一次突破（多：L <= lvl-brk），從那分鐘起 30 分鐘內（含該分鐘，共 30 根）第一根收盤收回（C >= lvl+rec）。
    回傳 (突破 slot, 訊號 slot) 或 None。訊號 slot 必須 <= end_slot（12:00 收盤）。"""
    s0 = None
    for s in range(R0, end_slot + 1):
        if np.isnan(Lr[s]):
            continue
        if (d == 1 and Lr[s] <= lvl - brk) or (d == -1 and Hr[s] >= lvl + brk):
            s0 = s; break
    if s0 is None:
        return None
    for t in range(s0, min(s0 + 29, end_slot) + 1):
        if np.isnan(Cr[t]):
            continue
        if (d == 1 and Cr[t] >= lvl + rec) or (d == -1 and Cr[t] <= lvl - rec):
            return s0, t
    return None


SKIP = {}   # 記錄被刪的天數等


def gen_S11(N_, es):
    out = []; roll = 0
    for i in range(N):
        if not R.TRAD[i]:
            continue
        lv = nday_level(i, N_)
        if lv is None:
            roll += 1; continue
        cands = []
        for d in (1, -1):
            lvl = lv["lo"] if d == 1 else lv["hi"]
            lag = lv["lag_lo"] if d == 1 else lv["lag_hi"]
            if lag < 3:
                continue
            ev = soup_event(R.L[i], R.H[i], R.C[i], d, lvl, 3.0, 0.5)
            if ev is None:
                continue
            s0, t = ev
            info = {"N日價位": lvl, "價位日": lv["lo_day"] if d == 1 else lv["hi_day"], "lag": lag,
                    "突破時間": hhmm(s0), "收回時間": hhmm(t), "收回收盤": R.C[i, t]}
            if es:
                if not R.ETRAD[i]:
                    continue
                le = nday_level(i, N_, es=True)
                if le is None:
                    continue
                el = le["lo"] if d == 1 else le["hi"]
                if (le["lag_lo"] if d == 1 else le["lag_hi"]) < 3:
                    continue
                ee = soup_event(R.EL[i], R.EH[i], R.EC[i], d, el, 0.75, 0.25)
                if ee is None or ee[1] > t:
                    continue
                info.update({"ES價位": el, "ES收回時間": hhmm(ee[1])})
            with np.errstate(all="ignore"):
                ext = np.nanmin(R.L[i, R0:t + 1]) if d == 1 else np.nanmax(R.H[i, R0:t + 1])
            stop = ext - 2 * d
            info["當日極端"] = ext
            cands.append(dict(i=i, d=d, kind="mkt", e0=t + 1, stop_raw=stop, tgt=None, info=info, known=t, sig=t))
        out += pick_first(cands)
    SKIP[f"S11_N{N_}_roll"] = roll
    return out


def pick_first(cands, key="sig"):
    """QC10：先出現的訊號優先；多空同分鐘 → 不做。"""
    if not cands:
        return []
    t = min(c[key] for c in cands)
    first = [c for c in cands if c[key] == t]
    if len({c["d"] for c in first}) > 1:
        return []
    return [first[0]]


# =====================================================================================
# S06 SMT
def gen_S06(breaker, window):
    out = []
    endmax = slot("10:59") if window == "AM" else slot("14:59")     # 確認那根最後一分鐘（收盤 11:00／15:00）
    for i in range(N):
        if not (R.TRAD[i] and R.ETRAD[i]):
            continue
        nq = day_bars(i, 5, a=R0, b=1320)
        es = day_bars(i, 5, es=True, a=R0, b=1320)
        esmap = {int(ts): k for k, ts in enumerate(es["ts"])}
        cands = []
        nb = len(nq["ts"])
        for k in range(nb):
            ts = int(nq["ts"][k])
            if ts < slot("09:45"):
                continue
            if nq["te"][k] > endmax:
                break
            ke = esmap.get(ts)
            if ke is None:
                continue
            with np.errstate(all="ignore"):
                nlo = np.nanmin(R.L[i, R0:ts]); nhi = np.nanmax(R.H[i, R0:ts])
                elo = np.nanmin(R.EL[i, R0:ts]); ehi = np.nanmax(R.EH[i, R0:ts])
            for d in (1, -1):
                if d == 1:
                    nq_brk = nq["L"][k] < nlo; es_brk = es["L"][ke] < elo
                else:
                    nq_brk = nq["H"][k] > nhi; es_brk = es["H"][ke] > ehi
                ok = (nq_brk and not es_brk) if breaker == "NQ" else (es_brk and not nq_brk)
                if not ok:
                    continue
                # 確認：k 之後 3 根 5 分 K（依時間 ts+5..ts+15）內 NQ 收盤突破 H[k]／L[k]
                for kk in range(k + 1, nb):
                    if nq["ts"][kk] > ts + 15:
                        break
                    if nq["te"][kk] > endmax:
                        break
                    if (d == 1 and nq["C"][kk] > nq["H"][k]) or (d == -1 and nq["C"][kk] < nq["L"][k]):
                        te = int(nq["te"][kk])
                        ext = nq["L"][k:kk + 1].min() if d == 1 else nq["H"][k:kk + 1].max()
                        info = {"k開始": hhmm(ts), "NQ極值sofar": nlo if d == 1 else nhi, "NQ_k": nq["L"][k] if d == 1 else nq["H"][k],
                                "ES極值sofar": elo if d == 1 else ehi, "ES_k": es["L"][ke] if d == 1 else es["H"][ke],
                                "確認根": hhmm(nq["ts"][kk]), "確認收盤": nq["C"][kk], "k高/低": nq["H"][k] if d == 1 else nq["L"][k]}
                        cands.append(dict(i=i, d=d, kind="mkt", e0=te + 1, stop_raw=ext - 3 * d, tgt=None, info=info,
                                          known=te, sig=te))
                        break
        out += pick_first(cands)
    return out


# =====================================================================================
# S03 PO3 / Judas
def first_open(i, a, n=15):
    for s in range(a, a + n):
        if np.isfinite(R.O[i, s]):
            return R.O[i, s], s
    return np.nan, -1


def gen_S03(anchor):
    out = []
    a_slot = slot("00:00") if anchor == "mid" else slot("08:30")
    for i in range(N):
        if not R.TRAD[i]:
            continue
        A, sa = first_open(i, a_slot)
        o930, s930 = first_open(i, R0)
        if not np.isfinite(A) or not np.isfinite(o930) or o930 == A:
            continue
        d = 1 if o930 > A else -1
        # 第一分鐘越過錨點 0.25 點
        m = None
        for s in range(R0, slot("11:00")):
            if np.isnan(R.L[i, s]):
                continue
            if (d == 1 and R.L[i, s] <= A - 0.25) or (d == -1 and R.H[i, s] >= A + 0.25):
                m = s; break
        if m is None:
            continue
        b0 = R0 + 5 * ((m - R0) // 5)
        bb = day_bars(i, 5, a=b0, b=slot("11:00"))
        hit = None
        for k in range(len(bb["ts"])):
            if (d == 1 and bb["C"][k] > A) or (d == -1 and bb["C"][k] < A):
                hit = k; break
        if hit is None:
            continue
        te = int(bb["te"][hit])
        with np.errstate(all="ignore"):
            ext = np.nanmin(R.L[i, R0:te + 1]) if d == 1 else np.nanmax(R.H[i, R0:te + 1])
        info = {"錨點A": A, "錨點時間": hhmm(sa), "O930": o930, "越過時間": hhmm(m), "確認根": hhmm(bb["ts"][hit]),
                "確認收盤": bb["C"][hit], "RTH極端": ext}
        out.append(dict(i=i, d=d, kind="mkt", e0=te + 1, stop_raw=ext - 3 * d, tgt=None, info=info, known=te, sig=te))
    return out


# =====================================================================================
# S02 ICT 2022 model
def sweeps_5m(bb, pools, buf, a_end):
    """回傳 [(k, d, pool_name, pool_px)]：同一根掃多個池取最極端。多方只看低點池（PDL/ONL），空方只看高點池。"""
    ev = []
    for k in range(len(bb["ts"])):
        if bb["te"][k] > a_end:
            break
        for d in (1, -1):
            best = None
            for nm, px in pools.items():
                if not np.isfinite(px):
                    continue
                if d == 1 and nm in ("PDL", "ONL") and bb["L"][k] <= px - buf and bb["C"][k] > px:
                    if best is None or px < best[1]:
                        best = (nm, px)
                if d == -1 and nm in ("PDH", "ONH") and bb["H"][k] >= px + buf and bb["C"][k] < px:
                    if best is None or px > best[1]:
                        best = (nm, px)
            if best:
                ev.append((k, d, best[0], best[1]))
    return ev


def gen_S02(entry, target, es, pen=PEN):
    out = []
    WEND = slot("11:29")          # 11:30 收盤＝11:29 那分鐘
    for i in range(N):
        if not R.TRAD[i]:
            continue
        lv = R.LV.loc[i]
        pools = {"PDH": lv.pdh, "PDL": lv.pdl, "ONH": lv.onh, "ONL": lv.onl}
        full = day_bars(i, 5, a=0, b=1440)            # 當天全部 5 分 K（含夜盤，用來算轉折）
        ph, pl = pivots(full["H"], full["L"], 2)
        rth0 = int(np.searchsorted(full["ts"], R0))
        ev = sweeps_5m({k: v[rth0:] for k, v in full.items()}, pools, 2.0, WEND)
        ev = [(k + rth0, d, nm, px) for k, d, nm, px in ev]
        if es:
            if not R.ETRAD[i]:
                continue
            el = R.ELV.loc[i]
            epools = {"PDH": el.pdh, "PDL": el.pdl, "ONH": el.onh, "ONL": el.onl}
            eb = day_bars(i, 5, es=True, a=R0, b=1440)
            eev = sweeps_5m(eb, epools, 0.5, WEND)
            ets = {}
            for k, d, nm, px in eev:
                ets.setdefault((d, nm), []).append(int(eb["te"][k]))
        setups = []
        for k, d, nm, px in ev:
            if es:
                lst = ets.get((d, nm), [])
                if not any(t <= full["te"][k] for t in lst):
                    continue
            # 第 k 根收盤時已確定的最近轉折（p+2 <= k）
            piv = ph if d == 1 else pl
            ps = [p for p in range(0, k - 1) if piv[p] and p + 2 <= k]
            if not ps:
                continue
            p = ps[-1]
            ref = full["H"][p] if d == 1 else full["L"][p]
            mss = None
            for kk in range(k + 1, min(k + 6, len(full["ts"]) - 1) + 1):
                if full["te"][kk] > WEND:
                    break
                if full["ts"][kk] > full["ts"][k] + 30:     # 6 根 5 分 K 的時間範圍
                    break
                if (d == 1 and full["C"][kk] > ref) or (d == -1 and full["C"][kk] < ref):
                    mss = kk; break
            if mss is None:
                continue
            setups.append(dict(k=k, d=d, nm=nm, px=px, p=p, ref=ref, mss=mss, te=int(full["te"][mss])))
        setups.sort(key=lambda z: z["te"])
        free_t = -1; done = False
        # 同一根 MSS 多空 → 都不做
        tes = {}
        for z in setups:
            tes.setdefault(z["te"], set()).add(z["d"])
        for z in setups:
            if done:
                break
            if z["te"] < free_t or len(tes[z["te"]]) > 1:
                continue
            k, d, mss = z["k"], z["d"], z["mss"]
            legx = full["L"][k] if d == 1 else full["H"][k]
            info = {"掃池": z["nm"], "池價": z["px"], "掃蕩根": hhmm(full["ts"][k]), "掃蕩極端": legx,
                    "轉折根": hhmm(full["ts"][z["p"]]) if full["ts"][z["p"]] >= 0 else str(full["ts"][z["p"]]),
                    "轉折價": z["ref"], "MSS根": hhmm(full["ts"][mss]), "MSS收盤": full["C"][mss]}
            if entry in ("top", "ce"):
                fv = None
                for j in range(k + 1, mss + 1):
                    if j < 2:
                        continue
                    if d == 1 and full["L"][j] > full["H"][j - 2] and full["L"][j] - full["H"][j - 2] >= 3:
                        fv = (full["H"][j - 2], full["L"][j], j)
                    if d == -1 and full["H"][j] < full["L"][j - 2] and full["L"][j - 2] - full["H"][j] >= 3:
                        fv = (full["H"][j], full["L"][j - 2], j)
                if fv is None:
                    continue
                lo_, hi_, j = fv
                P = (hi_ if d == 1 else lo_) if entry == "top" else (lo_ + hi_) / 2
                inval = lo_ if d == 1 else hi_
                info.update({"FVG下緣": lo_, "FVG上緣": hi_, "FVG第3根": hhmm(full["ts"][j])})
            else:
                if d == 1:
                    leg_hi = full["H"][k:mss + 1].max(); leg_lo = legx
                    P = leg_hi - 0.705 * (leg_hi - leg_lo); inval = leg_lo
                else:
                    leg_lo = full["L"][k:mss + 1].min(); leg_hi = legx
                    P = leg_lo + 0.705 * (leg_hi - leg_lo); inval = leg_hi
                info.update({"leg高": leg_hi, "leg低": leg_lo})
            P = round(P * 4) / 4 if entry != "top" else P     # 掛單價取到 tick
            stop_raw = legx - 3 * d
            q = qc2(P, stop_raw, d)
            a = z["te"] + 1; b = min(z["te"] + 60, WEND)
            # 失效：之後 5 分 K 收盤越過 inval → 該根結束後失效
            closes = {int(full["te"][kk]): full["C"][kk] for kk in range(mss + 1, len(full["ts"]))}
            cancel_t = None
            for tt, cc in sorted(closes.items()):
                if tt > b:
                    break
                if (d == 1 and cc < inval) or (d == -1 and cc > inval):
                    cancel_t = tt; break
            bb_ = b if cancel_t is None else min(b, cancel_t)
            if q is None:
                # 停損 >145：不掛單（沒有掛單存在，不阻擋之後的 setup）
                SKIP["S02_gt145"] = SKIP.get("S02_gt145", 0) + 1
                continue
            stop, sl = q
            s = lim_fill(i, d, P, a, bb_, pen)
            if s < 0:
                free_t = bb_ + 1
                continue
            info.update({"掛單價": P, "掛單起": hhmm(a), "成交": hhmm(s)})
            tgt = None
            if target == "pool":
                with np.errstate(all="ignore"):
                    hh = np.nanmax(R.H[i, R0:s]) if s > R0 else -np.inf
                    ll = np.nanmin(R.L[i, R0:s]) if s > R0 else np.inf
                ref_px = min(P, R.O[i, s]) if d == 1 else max(P, R.O[i, s])
                cand = []
                for nm2, px2 in pools.items():
                    if not np.isfinite(px2):
                        continue
                    if d == 1 and px2 > ref_px and hh < px2:
                        cand.append(px2)
                    if d == -1 and px2 < ref_px and ll > px2:
                        cand.append(px2)
                if cand:
                    tgt = min(cand) if d == 1 else max(cand)
                info["目標池"] = tgt
            out.append(dict(i=i, d=d, kind="lim", e=s, P=P, stop_raw=stop_raw, tgt=tgt, info=info, known=s - 1, sig=s))
            done = True
    return out


# =====================================================================================
# S10 Wyckoff
def gen_S10(confirm, vwap):
    out = []
    for i in range(N):
        if not R.TRAD[i] or not np.isfinite(R.ADR10[i]):
            continue
        bb = day_bars(i, 5, a=0, b=1440)
        vw = R.VW(i) if vwap == "on" else None
        nb = len(bb["ts"])
        cands = []
        for k in range(12, nb):
            if bb["ts"][k] < R0:
                continue
            if bb["te"][k] > slot("14:29"):
                break
            RHh = bb["H"][k - 12:k].max(); RLl = bb["L"][k - 12:k].min(); Rr = RHh - RLl
            if not (20 <= Rr <= 0.35 * R.ADR10[i]):
                continue
            for d in (1, -1):
                if d == 1:
                    ok = bb["L"][k] < RLl - 2 and bb["C"][k] > RLl
                else:
                    ok = bb["H"][k] > RHh + 2 and bb["C"][k] < RHh
                if not ok:
                    continue
                sigk = None
                if confirm == "now":
                    sigk = k
                else:
                    for kk in range(k + 1, min(k + 6, nb - 1) + 1):
                        if d == 1 and bb["L"][kk] > bb["L"][k] and bb["V"][kk] < bb["V"][k] and bb["C"][kk] > bb["O"][kk]:
                            sigk = kk; break
                        if d == -1 and bb["H"][kk] < bb["H"][k] and bb["V"][kk] < bb["V"][k] and bb["C"][kk] < bb["O"][kk]:
                            sigk = kk; break
                if sigk is None:
                    continue
                te = int(bb["te"][sigk])
                if te < slot("09:59") or te > slot("14:29"):
                    continue
                info = {"k根": hhmm(bb["ts"][k]), "RH": RHh, "RL": RLl, "R": Rr, "ADR10": R.ADR10[i], "k高/低": bb["L"][k] if d == 1 else bb["H"][k],
                        "k收": bb["C"][k], "訊號根": hhmm(bb["ts"][sigk])}
                if vw is not None:
                    v = vw[te]
                    if not ((d == 1 and bb["C"][sigk] > v) or (d == -1 and bb["C"][sigk] < v)):
                        continue
                    info["VWAP"] = v
                ext = bb["L"][k] if d == 1 else bb["H"][k]
                cands.append(dict(i=i, d=d, kind="mkt", e0=te + 1, stop_raw=ext - 3 * d, tgt=RHh if d == 1 else RLl,
                                  info=info, known=te, sig=te))
        out += pick_first(cands)
    return out


# =====================================================================================
# S01 Silver Bullet
def atr1m_day(i):
    """當天 slot 0 起 1 分 Wilder ATR14（只用有值的分鐘），回傳每個 slot 收盤時的 ATR（NaN 分鐘沿用上一值）。"""
    m = np.isfinite(R.C[i])
    idx = np.where(m)[0]
    a = wilder_atr(R.H[i, idx], R.L[i, idx], R.C[i, idx], 14)
    out = np.full(1440, np.nan); out[idx] = a
    return pd.Series(out).ffill().values


def gen_S01(window, filt, pen=PEN):
    out = []
    wa, wb = (slot("10:00"), slot("10:59")) if window == "AM" else (slot("14:00"), slot("14:59"))
    for i in range(N):
        if not R.TRAD[i]:
            continue
        atr = atr1m_day(i)
        vw = R.VW(i)
        Hh, Ll, Oo, Cc = R.H[i], R.L[i], R.O[i], R.C[i]
        found = None
        for k in range(wa, wb + 1):
            if np.isnan(Cc[k]) or np.isnan(Cc[k - 1]) or np.isnan(Cc[k - 2]):
                continue
            a = atr[k - 2]
            if not np.isfinite(a):
                continue
            for d in (1, -1):
                if d == 1:
                    ok = Ll[k] > Hh[k - 2] and Ll[k] - Hh[k - 2] >= 4 and (Cc[k - 1] - Oo[k - 1]) >= 1.5 * a
                else:
                    ok = Hh[k] < Ll[k - 2] and Ll[k - 2] - Hh[k] >= 4 and (Oo[k - 1] - Cc[k - 1]) >= 1.5 * a
                if not ok:
                    continue
                if filt == "sweep":
                    sw = False
                    for j in range(k - 2 - 9, k - 1):
                        with np.errstate(all="ignore"):
                            if d == 1 and Ll[j] < np.nanmin(Ll[j - 30:j]):
                                sw = True; break
                            if d == -1 and Hh[j] > np.nanmax(Hh[j - 30:j]):
                                sw = True; break
                    if not sw:
                        continue
                if filt == "vwap":
                    if not ((d == 1 and Cc[k] > vw[k]) or (d == -1 and Cc[k] < vw[k])):
                        continue
                if found is None:
                    found = []
                found.append((k, d))
            if found:
                break
        if not found or len({d for _, d in found}) > 1:
            continue
        k, d = found[0]
        P = Ll[k] if d == 1 else Hh[k]
        stop_raw = Ll[k - 2] - 1 if d == 1 else Hh[k - 2] + 1
        q = qc2(P, stop_raw, d)
        if q is None:
            continue
        s = lim_fill(i, d, P, k + 1, wb, pen)
        if s < 0:
            continue
        info = {"FVG第3根": hhmm(k), "FVG": (Hh[k - 2], Ll[k]) if d == 1 else (Hh[k], Ll[k - 2]), "ATR(k-2)": round(atr[k - 2], 2),
                "第2根實體": Cc[k - 1] - Oo[k - 1], "VWAP(k)": vw[k], "掛單": P, "成交": hhmm(s)}
        out.append(dict(i=i, d=d, kind="lim", e=s, P=P, stop_raw=stop_raw, tgt=None, info=info, known=s - 1, sig=s,
                        xs_struct=slot("11:59") if window == "AM" else R.EOD))
    return out


# =====================================================================================
# S07 SMC 順勢 OB（折價區）
def gen_S07(vwap, pen=PEN):
    out = []
    for i in range(N):
        if not (R.TRAD[i] and R.PREV_SAME[i]):
            continue
        j = R.g.prev_rth_day(i)
        rows = list(range(j, i + 1))
        S, T0 = seq(rows, R0)
        t0 = S["t"][0]
        b15 = bars(S["o"], S["h"], S["l"], S["c"], S["v"], S["t"], 15)
        b5 = bars(S["o"], S["h"], S["l"], S["c"], S["v"], S["t"], 5)
        n15 = len(b15["te"])
        ph, pl = pivots(b15["H"], b15["L"], 2)
        state = np.zeros(n15, int); rlo = np.full(n15, np.nan); rhi = np.full(n15, np.nan)
        lastSH = lastSL = None; st = 0
        for b in range(n15):
            # 可用的確定轉折（p+2 <= b-1）用來判斷本根收盤突破
            p = b - 3
            if p >= 0 and ph[p]:
                lastSH = p
            if p >= 0 and pl[p]:
                lastSL = p
            if lastSH is not None and b15["C"][b] > b15["H"][lastSH]:
                st = 1
            elif lastSL is not None and b15["C"][b] < b15["L"][lastSL]:
                st = -1
            state[b] = st
            # 區間：本根收盤時已確定（p+2 <= b）的最近轉折
            pp = b - 2
            cl = [q for q in range(0, pp + 1) if pl[q]] if st == 1 else [q for q in range(0, pp + 1) if ph[q]]
            if st != 0 and cl:
                q = cl[-1]
                if st == 1:
                    rlo[b] = b15["L"][q]; rhi[b] = b15["H"][q:b + 1].max()
                else:
                    rhi[b] = b15["H"][q]; rlo[b] = b15["L"][q:b + 1].min()
        # 5 分 OB
        n5 = len(b5["te"])
        p5h, p5l = pivots(b5["H"], b5["L"], 2)
        obs = []   # (d, top, bottom, formed_t, j)
        lastH = lastL = None; brokeH = brokeL = set()
        brokeH = set(); brokeL = set()
        for b in range(n5):
            p = b - 3
            if p >= 0 and p5h[p]:
                lastH = p
            if p >= 0 and p5l[p]:
                lastL = p
            if lastH is not None and lastH not in brokeH and b5["C"][b] > b5["H"][lastH]:
                brokeH.add(lastH)
                seg_ = b5["L"][lastH + 1:b + 1]
                m = lastH + 1 + (len(seg_) - 1 - int(np.argmin(seg_[::-1])))
                jj = [x for x in range(max(m - 3, 0), m + 1) if b5["C"][x] < b5["O"][x]]
                if jj:
                    x = jj[-1]
                    obs.append(dict(d=1, top=b5["O"][x], bot=b5["L"][x], ft=int(b5["te"][b]), jt=int(b5["ts"][x]), bt=int(b5["ts"][b])))
            if lastL is not None and lastL not in brokeL and b5["C"][b] < b5["L"][lastL]:
                brokeL.add(lastL)
                seg_ = b5["H"][lastL + 1:b + 1]
                m = lastL + 1 + (len(seg_) - 1 - int(np.argmax(seg_[::-1])))
                jj = [x for x in range(max(m - 3, 0), m + 1) if b5["C"][x] > b5["O"][x]]
                if jj:
                    x = jj[-1]
                    obs.append(dict(d=-1, top=b5["H"][x], bot=b5["O"][x], ft=int(b5["te"][b]), jt=int(b5["ts"][x]), bt=int(b5["ts"][b])))
        if not obs:
            continue
        # 每個 OB 的第一次觸碰時間、失效時間（全域分鐘）
        tt = S["t"]
        for ob in obs:
            after = tt > ob["ft"]
            hit = after & ((S["l"] <= ob["top"]) if ob["d"] == 1 else (S["h"] >= ob["bot"]))
            ob["touch"] = int(tt[np.argmax(hit)]) if hit.any() else 10 ** 9
            inv = (b5["te"] > ob["ft"]) & ((b5["C"] < ob["bot"]) if ob["d"] == 1 else (b5["C"] > ob["top"]))
            ob["inv"] = int(b5["te"][np.argmax(inv)]) if inv.any() else 10 ** 9
        vw = R.VW(i)
        for s in range(slot("09:45"), slot("14:30")):
            if NEWS_A <= s <= NEWS_B or np.isnan(R.H[i, s]):
                continue
            t = T0 + s
            b = int(np.searchsorted(b15["te"], t - 1, side="right")) - 1
            if b < 0 or state[b] == 0 or not np.isfinite(rlo[b]):
                continue
            d = state[b]
            el = [ob for ob in obs if ob["d"] == d and ob["ft"] <= t - 1 and ob["touch"] >= t and ob["inv"] >= t]
            if not el:
                continue
            ob = max(el, key=lambda z: z["ft"])
            eq = (rlo[b] + rhi[b]) / 2
            if (d == 1 and ob["top"] > eq) or (d == -1 and ob["bot"] < eq):
                continue
            if vwap == "on" and not ((d == 1 and R.C[i, s - 1] > vw[s - 1]) or (d == -1 and R.C[i, s - 1] < vw[s - 1])):
                continue
            P = ob["top"] if d == 1 else ob["bot"]
            stop_raw = ob["bot"] - 3 if d == 1 else ob["top"] + 3
            if qc2(P, stop_raw, d) is None:
                continue
            if (d == 1 and R.L[i, s] <= P - pen) or (d == -1 and R.H[i, s] >= P + pen):
                def tstr(x):
                    k = x - T0
                    return hhmm(k) if k >= 0 else "昨" + hhmm(k + 1440)
                info = {"15分狀態": d, "15分最後收盤根": tstr(int(b15["ts"][b])), "區間低": rlo[b], "區間高": rhi[b], "EQ": eq,
                        "OB": (ob["bot"], ob["top"]), "OB根": tstr(ob["jt"]), "BOS根": tstr(ob["bt"]), "OB確定": tstr(ob["ft"]), "成交": hhmm(s)}
                if vwap == "on":
                    info["VWAP(s-1)"] = vw[s - 1]; info["C(s-1)"] = R.C[i, s - 1]
                tg = rhi[b] if d == 1 else rlo[b]
                out.append(dict(i=i, d=d, kind="lim", e=s, P=P, stop_raw=stop_raw, tgt=tg, info=info, known=s - 1, sig=s))
                break
    return out


# =====================================================================================
# S09 MSNR
def edges60(nrows, start):
    """60 分 K 邊界（序列 index）。第一列從 start(930) 起。"""
    E = []
    for r in range(nrows):
        base = r * 1440 - start
        if r == 0:
            loc = list(range(930, 1380 + 1, 60))[:-1] + [1350, 1380, 1440]
        else:
            loc = list(range(0, 901, 60)) + [930, 990, 1050, 1110, 1170, 1230, 1290, 1350, 1380, 1440]
        loc = sorted(set(x for x in loc if x >= (start if r == 0 else 0)))
        E += [base + x for x in loc[:-1]]
    E.append(nrows * 1440 - start)
    return sorted(set(E))


def gen_S09(level_type, entry, pen=PEN):
    out = []
    for i in range(N):
        if not (R.TRAD[i] and R.PREV_SAME[i]):
            continue
        j1 = R.g.prev_rth_day(i); j2 = R.g.prev_rth_day(j1) if j1 is not None else None
        if j2 is None or R.CONTRACT[j2] != R.CONTRACT[i] or R.CONTRACT[j1] != R.CONTRACT[i]:
            SKIP["S09_roll"] = SKIP.get("S09_roll", 0) + 1
            continue
        rows = list(range(j2, i + 1))
        S, T0 = seq(rows, R0)
        tt = S["t"]; base = tt[0]
        b15 = bars(S["o"], S["h"], S["l"], S["c"], S["v"], tt, 15)
        atr15 = wilder_atr(b15["H"], b15["L"], b15["C"], 14)
        n15 = len(b15["te"])
        # 水平
        levs = []   # (X, type, init_role, confirm_t)
        cph, cpl = pivots(b15["C"], b15["C"], 2)
        for k in range(n15 - 2):
            if cph[k]:
                levs.append((b15["C"][k], "A", -1, int(b15["te"][k + 2])))
            if cpl[k]:
                levs.append((b15["C"][k], "V", 1, int(b15["te"][k + 2])))
        for k in range(1, n15):
            a = atr15[k - 1]
            if not np.isfinite(a):
                continue
            if b15["C"][k - 1] > b15["O"][k - 1] and b15["C"][k] > b15["O"][k] and b15["C"][k] - b15["O"][k] >= a:
                levs.append((b15["C"][k - 1], "G+", 1, int(b15["te"][k])))
            if b15["C"][k - 1] < b15["O"][k - 1] and b15["C"][k] < b15["O"][k] and b15["O"][k] - b15["C"][k] >= a:
                levs.append((b15["C"][k - 1], "G-", -1, int(b15["te"][k])))
        if not levs:
            continue
        # 60 分 storyline（A/V 水平）
        E = edges60(len(rows), R0)
        b60 = bars_custom(S["o"], S["h"], S["l"], S["c"], S["v"], tt, E)
        av = [(x, ty, ct) for x, ty, _, ct in levs if ty in ("A", "V")]
        sto = np.zeros(len(b60["te"]), int); cur = 0
        for b in range(1, len(b60["te"])):
            pc, c = b60["C"][b - 1], b60["C"][b]
            up = any(ty == "A" and ct < b60["ts"][b] and pc <= x < c for x, ty, ct in av)
            dn = any(ty == "V" and ct < b60["ts"][b] and pc >= x > c for x, ty, ct in av)
            if up and dn:
                cur = 0
            elif up:
                cur = 1
            elif dn:
                cur = -1
            sto[b] = cur
        use = [lv for lv in levs if (lv[1] in ("A", "V")) == (level_type == "AV")]
        if not use:
            continue
        X = np.array([u[0] for u in use]); IR = np.array([u[2] for u in use]); CT = np.array([u[3] for u in use])
        nl = len(use)
        # 角色：本根收盤後＝確定之後最後一個非零 (C15−X) 的符號，否則初始
        sg = np.sign(b15["C"][None, :] - X[:, None])
        sg = np.where(b15["te"][None, :] > CT[:, None], sg, 0)
        sgf = pd.DataFrame(np.where(sg == 0, np.nan, sg).T).ffill().values.T
        role = np.where(np.isnan(sgf), IR[:, None], sgf).astype(int)          # (nl, n15)
        prev = np.concatenate([IR[:, None], role[:, :-1]], 1)
        chg = (role != prev) & (b15["te"][None, :] > CT[:, None])
        rt = np.where(chg, b15["te"][None, :], -1)
        rt = np.maximum(np.maximum.accumulate(rt, 1), CT[:, None])           # 最後一次 refresh（或確定）時間
        # 觸碰（全域分鐘）
        M = len(tt)
        tch = (S["l"][None, :] <= X[:, None]) & (S["h"][None, :] >= X[:, None]) & (tt[None, :] > CT[:, None])
        lt = np.maximum.accumulate(np.where(tch, tt[None, :], -1), 1)        # 到該分鐘（含）最後一次觸碰
        vw = R.VW(i)

        def state_at(t):
            """t 分鐘開始時（只用 t-1 以前）：各水平 (存在, 角色, fresh)。"""
            b = int(np.searchsorted(b15["te"], t - 1, side="right")) - 1
            exist = CT <= t - 1
            if b >= 0:
                ro = role[:, b]; r_ = rt[:, b]
            else:
                ro = IR; r_ = CT
            r_ = np.where(exist, np.maximum(r_, CT), CT)
            ltt = lt[:, t - 1 - base] if t - 1 - base >= 0 else np.full(nl, -1)
            fresh = exist & (ltt <= r_)
            return exist, ro, fresh, b

        def story_at(t):
            b = int(np.searchsorted(b60["te"], t - 1, side="right")) - 1
            return sto[b] if b >= 0 else 0

        if entry == "limit":
            for s in range(R0, slot("15:00")):
                if NEWS_A <= s <= NEWS_B or np.isnan(R.H[i, s]):
                    continue
                t = T0 + s
                sy = story_at(t)
                if sy == 0:
                    continue
                exist, ro, fresh, b = state_at(t)
                if sy == 1:
                    m = exist & fresh & (ro == 1) & (R.L[i, s] <= X - pen)
                else:
                    m = exist & fresh & (ro == -1) & (R.H[i, s] >= X + pen)
                # 另一方向同分鐘也成交 → 不做
                m2 = exist & fresh & ((ro == -1) & (R.H[i, s] >= X + pen) if sy == 1 else (ro == 1) & (R.L[i, s] <= X - pen))
                if not m.any():
                    continue
                if m2.any():
                    break
                d = sy
                k = int(np.argmax(np.where(m, X, -np.inf))) if d == 1 else int(np.argmin(np.where(m, X, np.inf)))
                P = X[k]
                a15 = atr15[b] if b >= 0 else np.nan
                if not np.isfinite(a15):
                    break
                buf = np.ceil(max(15.0, 0.75 * a15) * 4) / 4      # 取到 tick（往外）
                stop_raw = P - d * buf
                if qc2(P, stop_raw, d) is None:
                    SKIP["S09_gt145"] = SKIP.get("S09_gt145", 0) + 1
                    continue
                info = {"水平": P, "類型": use[k][1], "確定": (hhmm(CT[k] - T0) if CT[k] >= T0 else f"前{(T0 - CT[k]) // 1440 + 1}日" + hhmm((CT[k] - T0) % 1440)),
                        "storyline": sy, "ATR15": round(a15, 1), "成交": hhmm(s)}
                out.append(dict(i=i, d=d, kind="lim", e=s, P=P, stop_raw=stop_raw, tgt=None, info=info, known=s - 1, sig=s))
                break
        else:
            b5 = day_bars(i, 5, a=R0, b=slot("15:00"))
            for kk in range(len(b5["ts"])):
                ts_, te_ = int(b5["ts"][kk]), int(b5["te"][kk])
                t = T0 + ts_
                exist, ro, fresh, b = state_at(t)
                sy = story_at(T0 + te_ + 1)
                if sy == 0:
                    continue
                l5, h5, c5 = b5["L"][kk], b5["H"][kk], b5["C"][kk]
                mL = exist & fresh & (ro == 1) & (l5 <= X) & (c5 > X)
                mS = exist & fresh & (ro == -1) & (h5 >= X) & (c5 < X)
                m = mL if sy == 1 else mS
                if not m.any():
                    continue
                d = sy
                k = int(np.argmax(np.where(m, X, -np.inf))) if d == 1 else int(np.argmin(np.where(m, X, np.inf)))
                stop_raw = l5 - 3 if d == 1 else h5 + 3
                info = {"水平": X[k], "類型": use[k][1], "確定": (hhmm(CT[k] - T0) if CT[k] >= T0 else f"前{(T0 - CT[k]) // 1440 + 1}日" + hhmm((CT[k] - T0) % 1440)),
                        "storyline": sy, "5分根": hhmm(ts_), "5分低/高": l5 if d == 1 else h5, "5分收": c5}
                out.append(dict(i=i, d=d, kind="mkt", e0=te_ + 1, stop_raw=stop_raw, tgt=None, info=info, known=te_, sig=te_))
                break
    return out
