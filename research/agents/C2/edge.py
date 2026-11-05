"""Inside-value-day VA-edge fade/acceptance signal generators (exploration; parameters free)."""
import numpy as np
from c2lib import RTH


def edge_events(g, lv, first_idx=4, last_idx=119, need_in=2, only_first=True):
    """Replicates the Pine '區邊回歸' reference logic and returns raw events with features.
    Inside-open days only. 5m closes from 09:34..; when a 5m close is outside VA, remember side;
    after `need_in` consecutive 5m closes back inside -> event (fade direction). One per day if only_first."""
    ev = []
    for r in lv.itertuples():
        if not r.has_prev or r.i2_dir != 0 or not np.isfinite(r.open_0930):
            continue
        i = r.i
        C = g.C[i, RTH:RTH + 390]; H = g.H[i, RTH:RTH + 390]; L = g.L[i, RTH:RTH + 390]
        side, cnt, xt, ext = 0, 0, None, np.nan
        nexc = 0
        for idx in range(first_idx, last_idx + 1, 5):
            c = C[idx]
            if not np.isfinite(c):
                continue
            if c > r.vah:
                if side != -1:
                    nexc += 1; xt = idx
                side, cnt = -1, 0
            elif c < r.val:
                if side != 1:
                    nexc += 1; xt = idx
                side, cnt = 1, 0
            elif side != 0:
                cnt += 1
                if cnt >= need_in:
                    hi = np.nanmax(H[:idx + 1]); lo = np.nanmin(L[:idx + 1])
                    ext = (hi - r.vah) if side == -1 else (r.val - lo)
                    ev.append({"i": i, "sig_idx": idx, "dir": side, "exc_idx": xt, "nexc": nexc,
                               "excursion": ext, "day_hi": hi, "day_lo": lo, "close": c})
                    if only_first:
                        break
                    side, cnt = 0, 0
    return ev


def edge_signals(g, lv, tp=55.0, sl=145.0, exit_idx=179, skip_news=True, last_entry=120, **kw):
    sigs = []
    lvi = lv.set_index("i")
    for e in edge_events(g, lv, **kw):
        ei = e["sig_idx"] + 1
        if (skip_news and 28 <= ei <= 32) or ei > last_entry or ei < 5:
            continue
        r = lvi.loc[e["i"]]
        s = {"i": e["i"], "entry_slot": RTH + ei, "dir": e["dir"], "exit_slot": RTH + exit_idx,
             "va_w": r.va_width, "adr": r.adr10, "open_pos": (r.open_0930 - r.val) / max(r.va_width, 1),
             "on_in": bool(r.onh <= r.vah and r.onl >= r.val),
             "exc": e["excursion"], "nexc": e["nexc"], "sig_idx": e["sig_idx"], "exc_idx": e["exc_idx"],
             "poc": r.poc, "vah": r.vah, "val": r.val, "close": e["close"], "day_hi": e["day_hi"], "day_lo": e["day_lo"]}
        # exits
        if tp == "poc":
            t = (e["close"] - r.poc) * (-e["dir"])  # approx distance from signal close to POC in trade direction
            s["tp_pts"] = max(abs(r.poc - e["close"]), 10.0) if (r.poc - e["close"]) * e["dir"] > 0 else 10.0
        elif tp == "opp":
            tgt = r.val if e["dir"] == -1 else r.vah
            s["tp_pts"] = max(abs(tgt - e["close"]), 10.0)
        else:
            s["tp_pts"] = tp
        if sl == "ext":   # stop beyond the excursion extreme + 5
            ex = e["day_hi"] if e["dir"] == -1 else e["day_lo"]
            s["sl_pts"] = abs(ex - e["close"]) + 5.0
        elif isinstance(sl, tuple) and sl[0] == "adr":
            s["sl_pts"] = sl[1] * r.adr10
        else:
            s["sl_pts"] = sl
        sigs.append(s)
    return sigs
