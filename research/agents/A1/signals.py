"""A1 signal generators (no look-ahead: decisions use closes up to entry_slot-1).
All return list of signal dicts for sim.run_signals."""
import numpy as np
from a1lib import RTH_OFF, sim

EOD = sim.slot_of("15:59")


def _first_true(mask_row):
    k = np.argmax(mask_row)
    return k if mask_row[k] else -1


def orb_levels(g, N):
    H = g.H[:, RTH_OFF:RTH_OFF + N]; L = g.L[:, RTH_OFF:RTH_OFF + N]
    with np.errstate(all="ignore"):
        return np.nanmax(H, 1), np.nanmin(L, 1)


def orb_break(g, lv, N=30, cutoff="12:00", exit_mode="or_eod", k=2.0, side="both", filt=None,
              tp=None, sl=None, exit_time="15:59", fade=False, days=None):
    """Opening-range breakout: first 1m close beyond OR after the OR completes.
    exit_mode:
      or_eod   : stop at opposite OR side, exit at exit_time
      r_k      : stop = OR range, target = k*OR range, exit at exit_time
      half_eod : stop = OR mid, exit at exit_time
      fixed    : tp/sl points, exit at exit_time
    fade=True: trade the failed breakout (close back inside OR after a break) in the opposite direction.
    """
    ORH, ORL = orb_levels(g, N)
    xs = sim.slot_of(exit_time); cs = sim.slot_of(cutoff)
    sigs = []
    for r in lv.itertuples():
        i = r.i
        if days is not None and not days[i]:
            continue
        if not r.tradable or not np.isfinite(ORH[i]):
            continue
        if filt is not None and not filt(i, r):
            continue
        h, l = ORH[i], ORL[i]; rng = h - l
        if rng <= 0:
            continue
        C = g.C[i, RTH_OFF + N - 1 + 1: cs]  # closes from first post-OR minute to cutoff-1
        if len(C) == 0:
            continue
        up = C > h; dn = C < l
        ku, kd = _first_true(up), _first_true(dn)
        cand = [(ku, 1), (kd, -1)]
        cand = [(k_, d_) for k_, d_ in cand if k_ >= 0]
        if not cand:
            continue
        k0, d0 = min(cand)
        base = RTH_OFF + N
        if fade:
            # after break at k0, wait for a close back inside the OR -> trade opposite
            inside = (C[k0 + 1:] <= h) & (C[k0 + 1:] >= l)
            kb = _first_true(inside) if len(inside) else -1
            if kb < 0:
                continue
            es = base + k0 + 1 + kb + 1
            d = -d0
            ext = np.nanmax(g.H[i, RTH_OFF:es]) if d0 == 1 else np.nanmin(g.L[i, RTH_OFF:es])
            stop_px = ext
            ref = g.O[i, es] if es < g.O.shape[1] else np.nan
            if not np.isfinite(ref):
                continue
            slp = abs(ref - stop_px) + 1.0
            if exit_mode == "fixed":
                s = {"tp_pts": tp, "sl_pts": sl}
            elif exit_mode == "r_k":
                s = {"tp_pts": k * slp, "sl_pts": slp}
            elif exit_mode == "target_other":   # target opposite OR side
                s = {"tp_pts": abs((l if d == -1 else h) - ref), "sl_pts": slp}
            else:
                s = {"tp_pts": None, "sl_pts": slp}
        else:
            es = base + k0 + 1
            d = d0
            if es >= cs + 1:
                continue
            ref = g.O[i, es]
            if not np.isfinite(ref):
                continue
            if exit_mode == "or_eod":
                s = {"tp_pts": None, "sl_pts": max(abs(ref - (l if d == 1 else h)), 1.0)}
            elif exit_mode == "half_eod":
                s = {"tp_pts": None, "sl_pts": max(abs(ref - (h + l) / 2), 1.0)}
            elif exit_mode == "r_k":
                s = {"tp_pts": k * rng, "sl_pts": rng}
            elif exit_mode == "fixed":
                s = {"tp_pts": tp, "sl_pts": sl}
            else:
                raise ValueError(exit_mode)
        if side == "long" and d != 1 or side == "short" and d != -1:
            continue
        s.update({"i": i, "entry_slot": es, "dir": d, "exit_slot": xs, "orr": rng})
        sigs.append(s)
    return sigs


def orb_candle(g, lv, N=5, stop="or", k=None, exit_time="15:59", filt=None, days=None):
    """Zarattini-Aziz style: direction of first N-min candle (close vs open), enter at 09:30+N,
    stop at the OR extreme (stop='or') or ATR fraction; target k*R or none; exit EOD."""
    ORH, ORL = orb_levels(g, N)
    xs = sim.slot_of(exit_time)
    sigs = []
    for r in lv.itertuples():
        i = r.i
        if days is not None and not days[i]:
            continue
        if not r.tradable or not np.isfinite(ORH[i]):
            continue
        if filt is not None and not filt(i, r):
            continue
        o = g.O[i, RTH_OFF]; c = g.C[i, RTH_OFF + N - 1]
        if not (np.isfinite(o) and np.isfinite(c)) or c == o:
            continue
        d = 1 if c > o else -1
        es = RTH_OFF + N
        ref = g.O[i, es]
        if not np.isfinite(ref):
            continue
        slp = max(abs(ref - (ORL[i] if d == 1 else ORH[i])), 1.0)
        s = {"i": i, "entry_slot": es, "dir": d, "exit_slot": xs, "sl_pts": slp,
             "tp_pts": (k * slp if k else None)}
        sigs.append(s)
    return sigs


def gap_signals(g, lv, d, mode="fill", lo=0.1, hi=0.5, ref="pdc", entry="09:30", stop_atr=0.5,
                target="ref", exit_time="15:59", confirm=None, days=None):
    """Gap trades. gap measured with the 09:29 close (known before 09:30 open) vs ref level
    (pdc | va: nearest VA edge | pdr: prior-day range edge | on: overnight range edge?).
    size in ATR units: lo <= |gap|/atr14 < hi.
    mode fill: trade toward ref; go: trade in gap direction.
    confirm: None or 'HH:MM' -> require the price at confirm-1 close to be beyond the 09:29 price in trade direction.
    target 'ref' (fill only: TP at ref level), None (EOD), or float = R multiple of stop.
    """
    es = sim.slot_of(entry); xs = sim.slot_of(exit_time)
    sigs = []
    for r in d.itertuples():
        i = r.i
        if days is not None and not days[i]:
            continue
        if not (r.tradable and r.has_prev) or not np.isfinite(r.atr14):
            continue
        p = g.C[i, RTH_OFF - 1]
        if not np.isfinite(p):
            continue
        if ref == "pdc":
            lvl = r.pdc
        elif ref == "va":
            lvl = r.vah if p > r.vah else (r.val if p < r.val else np.nan)
        elif ref == "pdr":
            lvl = r.pdh if p > r.pdh else (r.pdl if p < r.pdl else np.nan)
        if not np.isfinite(lvl):
            continue
        gap = p - lvl
        z = abs(gap) / r.atr14
        if not (lo <= z < hi) or gap == 0:
            continue
        dgap = 1 if gap > 0 else -1
        dr = -dgap if mode == "fill" else dgap
        e = es
        if confirm is not None:
            cs = sim.slot_of(confirm)
            c = g.C[i, cs - 1]
            if not np.isfinite(c) or (c - g.O[i, RTH_OFF]) * dr <= 0:
                continue
            e = cs
        px = g.O[i, e]
        if not np.isfinite(px):
            continue
        slp = stop_atr * r.atr14
        if target == "ref" and mode == "fill":
            tpp = (px - lvl) * dgap
            if tpp <= 0:
                continue
        elif target is None or target == "ref":
            tpp = None
        else:
            tpp = target * slp
        sigs.append({"i": i, "entry_slot": e, "dir": dr, "tp_pts": tpp, "sl_pts": slp, "exit_slot": xs, "gapz": z})
    return sigs


def noise_bands(g, lookback=14):
    """sigma[i, m] = mean over previous `lookback` tradable days of |C[m]/O930 - 1| (m = RTH minute 0..389)."""
    rC = g.rth(g.C); o = g.O[:, RTH_OFF]
    mv = np.abs(rC / o[:, None] - 1)
    sig = np.full(mv.shape, np.nan)
    hist = []
    for i in range(len(g)):
        if len(hist) >= lookback:
            sig[i] = np.nanmean(np.stack(hist[-lookback:]), 0)
        if g.tradable[i] and np.isfinite(o[i]):
            hist.append(mv[i])
    return sig


def noise_breakout(g, lv, sig, mult=1.0, first="10:00", last="15:00", step=30, exit_inside=True,
                   exit_time="15:59", stop_mode="band", days=None, filt=None):
    """Zarattini-style noise-area breakout, one trade per day. Checks at half-hour marks
    (close of the minute before the mark); entry at the mark's open. Hard stop at the opposite?/same band
    distance; optional exit at first later mark whose prior close is back inside the band (rule-based)."""
    xs = sim.slot_of(exit_time)
    marks = list(range(sim.slot_of(first), sim.slot_of(last) + 1, step))
    sigs = []
    for r in lv.itertuples():
        i = r.i
        if days is not None and not days[i]:
            continue
        if not (r.tradable and r.has_prev) or not np.isfinite(sig[i, 0]):
            continue
        if filt is not None and not filt(i, r):
            continue
        o = r.open_0930
        ub_base, lb_base = max(o, r.pdc), min(o, r.pdc)
        hit = None
        for mk in marks:
            m = mk - 1 - RTH_OFF
            c = g.C[i, mk - 1]
            if not np.isfinite(c):
                continue
            ub = ub_base * (1 + mult * sig[i, m]); lb = lb_base * (1 - mult * sig[i, m])
            if c > ub:
                hit = (mk, 1, ub); break
            if c < lb:
                hit = (mk, -1, lb); break
        if hit is None:
            continue
        mk, dr, band = hit
        px = g.O[i, mk]
        if not np.isfinite(px):
            continue
        slp = max(abs(px - band), 5.0) if stop_mode == "band" else None
        ex = xs
        if exit_inside:
            for mk2 in range(mk + step, xs + 1, step):
                m2 = mk2 - 1 - RTH_OFF
                c2 = g.C[i, mk2 - 1]
                if not np.isfinite(c2) or m2 >= 390:
                    continue
                ub = ub_base * (1 + mult * sig[i, m2]); lb = lb_base * (1 - mult * sig[i, m2])
                if (dr == 1 and c2 < ub) or (dr == -1 and c2 > lb):
                    ex = mk2 - 1; break
        sigs.append({"i": i, "entry_slot": mk, "dir": dr, "tp_pts": None, "sl_pts": slp, "exit_slot": ex})
    return sigs


def trend_afternoon(g, lv, d, at="12:00", thr=0.5, exit_time="15:59", stop_atr=0.3, days=None):
    """At `at`, if (close - 09:30 open)/ATR14 beyond thr, trade in that direction to the close."""
    es = sim.slot_of(at); xs = sim.slot_of(exit_time)
    sigs = []
    for r in d.itertuples():
        i = r.i
        if days is not None and not days[i]:
            continue
        if not r.tradable or not np.isfinite(r.atr14):
            continue
        c = g.C[i, es - 1]
        if not np.isfinite(c):
            continue
        z = (c - r.open_0930) / r.atr14
        if abs(z) < thr:
            continue
        sigs.append({"i": i, "entry_slot": es, "dir": 1 if z > 0 else -1, "tp_pts": None,
                     "sl_pts": stop_atr * r.atr14, "exit_slot": xs})
    return sigs


def level_break(g, lv, d, level="pd", mode="go", start="09:35", cutoff="13:00", stop_atr=0.25, exit_time="15:59",
                only_inside_open=True, days=None):
    """First 1m close beyond prior-day (pd) or overnight (on) high/low after `start`.
    mode go: continuation; fade: wait for close back inside then trade toward the range."""
    ss, cs, xs = sim.slot_of(start), sim.slot_of(cutoff), sim.slot_of(exit_time)
    sigs = []
    for r in d.itertuples():
        i = r.i
        if days is not None and not days[i]:
            continue
        if not (r.tradable and r.has_prev) or not np.isfinite(r.atr14):
            continue
        hi, lo = (r.pdh, r.pdl) if level == "pd" else (r.onh, r.onl)
        o = r.open_0930
        if only_inside_open and not (lo <= o <= hi):
            continue
        C = g.C[i, ss - 1: cs]
        ku, kd = _first_true(C > hi), _first_true(C < lo)
        cand = [(k, dd) for k, dd in [(ku, 1), (kd, -1)] if k >= 0]
        if not cand:
            continue
        k0, d0 = min(cand)
        if mode == "go":
            es, dr = ss - 1 + k0 + 1, d0
        else:
            lvl = hi if d0 == 1 else lo
            back = (C[k0 + 1:] < hi) if d0 == 1 else (C[k0 + 1:] > lo)
            kb = _first_true(back) if len(back) else -1
            if kb < 0:
                continue
            es, dr = ss - 1 + k0 + 1 + kb + 1, -d0
        if es > cs or not np.isfinite(g.O[i, es]):
            continue
        sigs.append({"i": i, "entry_slot": es, "dir": dr, "tp_pts": None, "sl_pts": stop_atr * r.atr14, "exit_slot": xs})
    return sigs
