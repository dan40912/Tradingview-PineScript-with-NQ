"""B1 indicator library: volume / market-profile / order-flow-proxy features for NQ (or ES).

Reusable by other agents:
    import sys; sys.path.insert(0, "research"); sys.path.insert(0, "research/agents/B1")
    from lib import data, features
    import indicators as ind
    g = data.load_grid("NQ"); lv = features.daily_levels(g)
    pf  = ind.profile_features(g, lv)          # one row per day i (pre-09:30 info only)
    ia  = ind.intraday_arrays(g)               # per-minute arrays (ndays, 1440): vwap, bands, CLV delta
    dva = ind.developing_va(g, i, upto_slot)   # (val, poc, vah) of RTH so far (bars < upto_slot)

NO LOOK-AHEAD CONVENTIONS
* Every column in profile_features() uses only data that closed before 09:30 ET of day i
  (prior RTH sessions + today's overnight slots 0..929), except the columns prefixed `or_`
  (opening-range: they use slots 930..930+N-1 and are only valid for decisions at/after 09:30+N).
* intraday_arrays()[k][i, s] uses bars with slot <= s (value known at the CLOSE of minute s,
  i.e. usable for an entry at slot s+1).
* POC levels returned here are bin MIDPOINTS (poc_lo + 2.5).  The frozen lv.poc is the bin lower edge;
  history showed a POC signal that vanished when moving 2.5 pts, so midpoints are used deliberately.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

RTH_OFF = 930
BIN = 5.0


# ----------------------------------------------------------------------------- profiles
def profile(h, l, v, bin_=BIN):
    """Volume profile with the frozen I2 spreading rule. Returns (kmin, prof) where bin k covers
    [k*bin, (k+1)*bin). Empty -> (None, None)."""
    m = np.isfinite(v) & np.isfinite(h) & np.isfinite(l)
    h, l, v = h[m], l[m], v[m]
    if len(v) == 0:
        return None, None
    a = np.floor(l / bin_).astype(int)
    b = np.floor(h / bin_).astype(int)
    kmin, kmax = a.min(), b.max()
    prof = np.zeros(kmax - kmin + 2)
    w = v / (b - a + 1)
    # difference-array trick: +w at a, -w at b+1
    np.add.at(prof, a - kmin, w)
    np.add.at(prof, b - kmin + 1, -w)
    return kmin, np.cumsum(prof)[:-1]


def va_from_profile(kmin, prof, frac=0.70, bin_=BIN):
    """Frozen expansion rule. Returns (val, poc_lo, vah) with poc_lo = POC bin lower edge."""
    if prof is None or prof.sum() <= 0:
        return (np.nan,) * 3
    total = prof.sum()
    p = int(np.argmax(prof))
    lo = hi = p
    acc = prof[p]
    n = len(prof)
    while acc < frac * total and (lo > 0 or hi < n - 1):
        lvv = prof[lo - 1] if lo > 0 else -1.0
        hvv = prof[hi + 1] if hi < n - 1 else -1.0
        if hvv > lvv:
            hi += 1; acc += max(0.0, hvv)
        else:
            lo -= 1; acc += max(0.0, lvv)
    return (lo + kmin) * bin_, (p + kmin) * bin_, (hi + kmin + 1) * bin_


def add_profiles(parts):
    """Sum a list of (kmin, prof) on a common axis."""
    parts = [p for p in parts if p[0] is not None]
    if not parts:
        return None, None
    k0 = min(p[0] for p in parts)
    k1 = max(p[0] + len(p[1]) for p in parts)
    out = np.zeros(k1 - k0)
    for k, pr in parts:
        out[k - k0:k - k0 + len(pr)] += pr
    return k0, out


def tpo_counts(h, l, period=30, bin_=BIN):
    """TPO profile: for each 5-pt bin, number of `period`-minute brackets that traded it.
    h, l = 1-min RTH arrays (390). Returns (kmin, counts)."""
    nper = int(np.ceil(len(h) / period))
    lows, highs = [], []
    for p in range(nper):
        hh, ll = h[p * period:(p + 1) * period], l[p * period:(p + 1) * period]
        if np.isfinite(hh).any():
            highs.append(np.nanmax(hh)); lows.append(np.nanmin(ll))
    if not highs:
        return None, None
    a = np.floor(np.array(lows) / bin_).astype(int)
    b = np.floor(np.array(highs) / bin_).astype(int)
    kmin = a.min()
    cnt = np.zeros(b.max() - kmin + 1, int)
    for ai, bi in zip(a, b):
        cnt[ai - kmin:bi - kmin + 1] += 1
    return kmin, cnt


def developing_va(g, i, upto_slot, start_slot=RTH_OFF, frac=0.70):
    """Value area of bars start_slot .. upto_slot-1 (known at open of upto_slot)."""
    k, pr = profile(g.H[i, start_slot:upto_slot], g.L[i, start_slot:upto_slot], g.V[i, start_slot:upto_slot])
    return va_from_profile(k, pr, frac)


# ----------------------------------------------------------------------------- daily features
def _rth_profiles(g):
    rH, rL, rV = g.rth(g.H), g.rth(g.L), g.rth(g.V)
    return [profile(rH[i], rL[i], rV[i]) if g.has_rth[i] else (None, None) for i in range(len(g))]


def _prev_rth_list(g, i, n):
    """Up to n previous RTH day indices with the same contract as day i (most recent first)."""
    out, j = [], i
    while len(out) < n:
        j = g.prev_rth_day(j)
        if j is None or g.contract[j] != g.contract[i]:
            break
        out.append(j)
    return out


def profile_features(g, lv, ndays_rvol=20):
    """One row per day i. Everything uses data available before 09:30 of day i unless prefixed `or_`."""
    n = len(g)
    profs = _rth_profiles(g)
    rH, rL, rC, rV, rO = g.rth(g.H), g.rth(g.L), g.rth(g.C), g.rth(g.V), g.rth(g.O)
    rth_range = np.array([np.nanmax(rH[i]) - np.nanmin(rL[i]) if g.has_rth[i] else np.nan for i in range(n)])
    rth_vol = np.array([np.nansum(rV[i]) if g.has_rth[i] else np.nan for i in range(n)])
    # VA per day (frozen) + midpoint POC
    va = [va_from_profile(*profs[i]) for i in range(n)]
    # opening-window volumes for RVOL
    ow = {m: np.nansum(rV[:, :m], axis=1) for m in (5, 15, 30)}
    on_v = np.nansum(g.V[:, :RTH_OFF], axis=1)
    # full-session hi/lo segments for naked-POC tracking
    post_hi = np.nanmax(np.where(np.isfinite(g.H[:, 1320:]), g.H[:, 1320:], -np.inf), axis=1)
    post_lo = np.nanmin(np.where(np.isfinite(g.L[:, 1320:]), g.L[:, 1320:], np.inf), axis=1)
    full_hi = np.nanmax(np.where(np.isfinite(g.H), g.H, -np.inf), axis=1)
    full_lo = np.nanmin(np.where(np.isfinite(g.L), g.L, np.inf), axis=1)
    on_hi = np.nanmax(np.where(np.isfinite(g.H[:, :RTH_OFF]), g.H[:, :RTH_OFF], -np.inf), axis=1)
    on_lo = np.nanmin(np.where(np.isfinite(g.L[:, :RTH_OFF]), g.L[:, :RTH_OFF], np.inf), axis=1)

    rows = []
    for i in range(n):
        d = {"i": i}
        r = lv.iloc[i]
        o = r.open_0930
        prev = _prev_rth_list(g, i, 20)
        rth_prev_all = []
        j = i
        while len(rth_prev_all) < ndays_rvol:   # any contract, for volume / range normalisation
            j = g.prev_rth_day(j)
            if j is None:
                break
            if g.tradable[j]:
                rth_prev_all.append(j)
        # ---- ATR-like normaliser: mean RTH range of last 14 tradable sessions
        atr = np.nanmean(rth_range[rth_prev_all[:14]]) if len(rth_prev_all) >= 5 else np.nan
        d["atr14"] = atr
        # ---- relative volume (pre-open and opening windows) vs prior 20 tradable sessions
        if len(rth_prev_all) >= 10:
            d["rvol_prev_rth"] = rth_vol[rth_prev_all[0]] / np.nanmean(rth_vol[rth_prev_all])
            d["rvol_on"] = on_v[i] / np.nanmean(on_v[rth_prev_all])
            for m in (5, 15, 30):
                d[f"or_rvol{m}"] = ow[m][i] / np.nanmean(ow[m][rth_prev_all])
        else:
            d["rvol_prev_rth"] = d["rvol_on"] = np.nan
            for m in (5, 15, 30):
                d[f"or_rvol{m}"] = np.nan
        if not r.has_prev or not prev:
            rows.append(d); continue
        j = prev[0]
        pdh, pdl = np.nanmax(rH[j]), np.nanmin(rL[j])
        rng = pdh - pdl
        val, poclo, vah = va[j]
        poc = poclo + BIN / 2
        d.update(pd_range=rng, va_w_atr=(vah - val) / atr if atr else np.nan,
                 va_w_rng=(vah - val) / rng if rng > 0 else np.nan,
                 poc_mid=poc, poc_pos=(poc - pdl) / rng if rng > 0 else np.nan,
                 va_mid_pos=((val + vah) / 2 - pdl) / rng if rng > 0 else np.nan,
                 rng_atr=rng / atr if atr else np.nan)
        d["shape"] = "P" if d["poc_pos"] > 2 / 3 else ("b" if d["poc_pos"] < 1 / 3 else "D")
        # close location in prior range / vs VA
        last = np.where(np.isfinite(rC[j]))[0]
        pdc = rC[j, last[-1]]
        d["pd_close_pos"] = (pdc - pdl) / rng if rng > 0 else np.nan
        # ---- poor highs / lows and single prints (30-min TPO, 5-pt bins)
        kt, cnt = tpo_counts(rH[j], rL[j])
        if cnt is not None:
            d["tpo_top"] = int(cnt[-1]); d["tpo_bot"] = int(cnt[0])
            d["poor_high"] = int(cnt[-1] >= 2); d["poor_low"] = int(cnt[0] >= 2)
            # tails = consecutive single-TPO bins from each extreme
            t_hi = 0
            while t_hi < len(cnt) and cnt[-1 - t_hi] == 1:
                t_hi += 1
            t_lo = 0
            while t_lo < len(cnt) and cnt[t_lo] == 1:
                t_lo += 1
            d["tail_hi_pts"] = t_hi * BIN; d["tail_lo_pts"] = t_lo * BIN
            inner = cnt[t_lo:len(cnt) - t_hi] if t_hi + t_lo < len(cnt) else cnt[:0]
            d["single_prints_pts"] = int((inner == 1).sum()) * BIN
            # volume-based poor extremes: top/bottom 5-pt bin volume relative to mean bin
            kp, pr = profs[j]
            d["vol_top_rel"] = pr[-1] / pr.mean(); d["vol_bot_rel"] = pr[0] / pr.mean()
        # ---- VA migration vs the session before
        if len(prev) >= 2:
            v2, p2, h2 = va[prev[1]]
            mig = 2 if val > h2 else (-2 if vah < v2 else (1 if poclo > p2 else (-1 if poclo < p2 else 0)))
            d["va_migration"] = mig   # +2 higher (no overlap), +1 overlapping-higher, 0, -1, -2
            d["poc_shift_atr"] = (poclo - p2) / atr if atr else np.nan
        # ---- composite (multi-day) value areas
        for k in (2, 3, 5):
            if len(prev) >= k:
                kc, pc = add_profiles([profs[x] for x in prev[:k]])
                cv, cp, ch = va_from_profile(kc, pc)
                d[f"c{k}_val"], d[f"c{k}_poc"], d[f"c{k}_vah"] = cv, cp + BIN / 2, ch
                d[f"c{k}_open_pos"] = (o - cv) / (ch - cv) if ch > cv else np.nan  # <0 below, >1 above
        # ---- naked POCs (same contract, ages 2..20 sessions back; untouched through 09:29 today)
        nps = []
        hi_since, lo_since = max(post_hi[j], on_hi[i]), min(post_lo[j], on_lo[i])
        # walk back: for day prev[a], 'since' covers its post-RTH, all full days after it, today's ON
        for a in range(1, len(prev)):
            k_ = prev[a]
            nxt = prev[a - 1]
            hi_since = max(hi_since, full_hi[nxt]); lo_since = min(lo_since, full_lo[nxt])
            # also any non-RTH days between k_ and nxt
            for z in range(k_ + 1, nxt):
                hi_since = max(hi_since, full_hi[z]); lo_since = min(lo_since, full_lo[z])
            pk = va[k_][1] + BIN / 2
            hk = max(hi_since, post_hi[k_])
            lk = min(lo_since, post_lo[k_])
            if not (lk <= pk <= hk):
                nps.append(pk)
        nps = np.array(nps)
        d["n_naked_poc"] = len(nps)
        if len(nps):
            up, dn = nps[nps > o], nps[nps < o]
            d["naked_up_dist"] = (up.min() - o) if len(up) else np.nan
            d["naked_dn_dist"] = (o - dn.max()) if len(dn) else np.nan
        d["prev_poc_naked"] = int(not (min(post_lo[j], on_lo[i]) <= poc <= max(post_hi[j], on_hi[i])))
        # ---- overnight (ETH) profile
        ko, po = profile(g.H[i, :RTH_OFF], g.L[i, :RTH_OFF], g.V[i, :RTH_OFF])
        if po is not None:
            oval, opoc, ovah = va_from_profile(ko, po)
            d.update(on_val=oval, on_poc=opoc + BIN / 2, on_vah=ovah)
            d["on_va_rel"] = 2 if oval > vah else (-2 if ovah < val else (1 if opoc > poclo else (-1 if opoc < poclo else 0)))
            d["open_vs_onva"] = 1 if o > ovah else (-1 if o < oval else 0)
            d["on_range_atr"] = (on_hi[i] - on_lo[i]) / atr if atr else np.nan
            d["on_inside_pdva"] = int(on_hi[i] <= vah and on_lo[i] >= val)
        # ---- overnight CLV-delta proxy (sum clv*v over ETH) normalised by ETH volume
        H_, L_, C_, V_ = g.H[i, :RTH_OFF], g.L[i, :RTH_OFF], g.C[i, :RTH_OFF], g.V[i, :RTH_OFF]
        rr = H_ - L_
        clv = np.where(rr > 0, ((C_ - L_) - (H_ - C_)) / np.where(rr > 0, rr, 1), 0.0)
        m = np.isfinite(clv) & np.isfinite(V_)
        d["on_delta_frac"] = np.sum(clv[m] * V_[m]) / max(np.sum(V_[m]), 1)
        # prior RTH session delta proxy
        H_, L_, C_, V_ = rH[j], rL[j], rC[j], rV[j]
        rr = H_ - L_
        clv = np.where(rr > 0, ((C_ - L_) - (H_ - C_)) / np.where(rr > 0, rr, 1), 0.0)
        m = np.isfinite(clv) & np.isfinite(V_)
        d["pd_delta_frac"] = np.sum(clv[m] * V_[m]) / max(np.sum(V_[m]), 1)
        rows.append(d)
    return pd.DataFrame(rows).set_index("i").reindex(range(n))


# ----------------------------------------------------------------------------- intraday arrays
def intraday_arrays(g):
    """Per-minute arrays (ndays, 1440), value at [i, s] known at the close of slot s.
    vwap18/sd18: VWAP & vol-weighted std of typical price anchored at 18:00 (slot 0)
    vwap930/sd930: anchored at 09:30 (NaN before 09:30)
    clv: close-location value of each bar in [-1, 1]; dlt = clv*volume (delta proxy)
    cd18 / cd930: cumulative delta proxy from 18:00 / 09:30
    """
    H, L, C, V = g.H, g.L, g.C, g.V
    tp = (H + L + C) / 3
    vv = np.where(np.isfinite(V) & np.isfinite(tp), V, 0.0)
    tpz = np.where(np.isfinite(tp), tp, 0.0)
    rr = H - L
    clv = np.where(rr > 0, ((C - L) - (H - C)) / np.where(rr > 0, rr, 1.0), 0.0)
    clv = np.where(np.isfinite(clv), clv, 0.0)
    dlt = clv * vv
    out = {"clv": clv, "dlt": dlt}

    def anchored(start):
        w = vv.copy(); w[:, :start] = 0
        cw = np.cumsum(w, 1)
        cpv = np.cumsum(w * tpz, 1)
        cp2 = np.cumsum(w * tpz * tpz, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            vw = cpv / cw
            sd = np.sqrt(np.maximum(cp2 / cw - vw * vw, 0))
        vw[:, :start] = np.nan; sd[:, :start] = np.nan
        vw[cw == 0] = np.nan; sd[cw == 0] = np.nan
        return vw, sd

    out["vwap18"], out["sd18"] = anchored(0)
    out["vwap930"], out["sd930"] = anchored(RTH_OFF)
    out["cd18"] = np.cumsum(dlt, 1)
    d2 = dlt.copy(); d2[:, :RTH_OFF] = 0
    out["cd930"] = np.cumsum(d2, 1); out["cd930"][:, :RTH_OFF] = np.nan
    return out


def developing_va_path(g, i, start_slot=RTH_OFF, end_slot=RTH_OFF + 180, step=5, frac=0.70):
    """Developing VA sampled every `step` minutes: dict slot -> (val, poc_mid, vah) using bars < slot."""
    res = {}
    for s in range(start_slot + step, end_slot + 1, step):
        a, p, b = developing_va(g, i, s, start_slot, frac)
        res[s] = (a, p + BIN / 2 if np.isfinite(p) else np.nan, b)
    return res


# ----------------------------------------------------------------------------- ES relation
def es_features(g, ges, lv_es):
    """Per-day NQ-vs-ES features known at 09:30 open (uses slot 929 closes) and ES VA position."""
    n = len(g)
    rows = []
    for i in range(n):
        d = {"i": i}
        r = lv_es.iloc[i]
        if g.dates[i] != ges.dates[i]:
            rows.append(d); continue
        o_es = r.open_0930
        d["es_dir"] = int(r.i2_dir)
        w = r.vah - r.val
        d["es_open_va_z"] = ((o_es - r.vah) / w if o_es > r.vah else ((o_es - r.val) / w if o_es < r.val else 0.0)) if w > 0 else np.nan
        # overnight returns (18:00 open -> 09:29 close) relative
        def ret(G):
            o = G.O[i, :RTH_OFF]; c = G.C[i, :RTH_OFF]
            fo = o[np.isfinite(o)]; fc = c[np.isfinite(c)]
            return (fc[-1] / fo[0] - 1) if len(fo) and len(fc) else np.nan
        d["nq_on_ret"] = ret(g); d["es_on_ret"] = ret(ges)
        d["rs_on"] = d["nq_on_ret"] - d["es_on_ret"]
        rows.append(d)
    return pd.DataFrame(rows).set_index("i").reindex(range(n))
