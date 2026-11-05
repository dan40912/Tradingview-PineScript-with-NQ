"""Liquidity sweep + reclaim signal generator (no look-ahead).
A level (overnight high/low, prior-day high/low, or a session range) is 'swept' when a 1m high (low)
trades >= buf points beyond it after 09:30; a 'reclaim' is a 5-minute close (minute idx%5==4, same clock as I2)
back on the inside of the level, no later than `last_idx`.  Entry = open of the next minute.
Stop = sweep extreme (highest high / lowest low from 09:30 to signal bar) + stop_buf, optionally capped/floored.
Target = R-multiple of the risk, or a price level (e.g. POC / range mid), else time exit."""
import numpy as np
from lib.data import RTH_OFF


def sweep_signals(g, lv, level="on", buf=2.0, last_idx=120, stop_buf=5.0, min_risk=10.0, max_risk=80.0,
                  tgt="R", R=2.0, exit_idx=389, need_open_inside=True, sides=(1, -1), extra_filter=None,
                  session=None):
    sigs = []
    for r in lv.itertuples():
        if not r.has_prev or not np.isfinite(r.open_0930):
            continue
        i = r.i
        if level == "on":
            hi, lo = r.onh, r.onl
        elif level == "pd":
            hi, lo = r.pdh, r.pdl
        elif level == "va":
            hi, lo = r.vah, r.val
        elif level == "sess":
            hi, lo = session[i]
        else:
            raise ValueError(level)
        if not (np.isfinite(hi) and np.isfinite(lo)):
            continue
        o = r.open_0930
        if need_open_inside and not (lo < o < hi):
            continue
        H = g.H[i, RTH_OFF:RTH_OFF + 390]; L = g.L[i, RTH_OFF:RTH_OFF + 390]; C = g.C[i, RTH_OFF:RTH_OFF + 390]
        swept_hi = swept_lo = False
        xh, xl = -np.inf, np.inf
        for idx in range(0, last_idx + 1):
            h, l = H[idx], L[idx]
            if np.isfinite(h):
                xh = max(xh, h); xl = min(xl, l)
                if h >= hi + buf: swept_hi = True
                if l <= lo - buf: swept_lo = True
            if idx % 5 != 4:
                continue
            c = C[idx]
            if not np.isfinite(c):
                continue
            d = 0
            if -1 in sides and swept_hi and c < hi:
                d = -1
            elif 1 in sides and swept_lo and c > lo:
                d = 1
            if d == 0:
                continue
            if idx + 1 > last_idx or 28 <= idx + 1 <= 32:   # skip 09:58-10:02 like I2
                # still one decision per day: first reclaim decides
                break
            if extra_filter is not None and not extra_filter(r, d, idx, c):
                break
            ext = xh if d == -1 else xl
            risk = abs(ext - c) + stop_buf
            if risk < min_risk: risk = min_risk
            if max_risk is not None and risk > max_risk:
                break
            if tgt == "R":
                tp = R * risk
            elif tgt == "mid":
                tp = abs((hi + lo) / 2 - c)
            elif tgt == "poc":
                tp = abs(r.poc - c) if (r.poc - c) * d > 0 else None
            elif tgt == "opp":
                tp = abs((lo if d == -1 else hi) - c)
            elif tgt == "none":
                tp = np.inf
            if tp is None or tp < 5:
                break
            sigs.append({"i": i, "entry_slot": RTH_OFF + idx + 1, "dir": d, "tp_pts": tp, "sl_pts": risk,
                         "exit_slot": RTH_OFF + exit_idx, "tag": f"sweep_{level}"})
            break
    return sigs
