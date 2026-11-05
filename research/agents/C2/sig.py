"""C2 exploration signal generators (free parameters). Every decision uses closes <= entry_slot-1,
prior-day levels, the 09:30 open, and I2 signals already issued before the decision minute."""
import numpy as np
from c2lib import RTH
from lib import strategies


def i2_entry_idx(g, lv):
    """{day i: I2 v1 entry idx (minutes since 09:30)} -- the I2 signal is known in real time at entry-1."""
    return {s["i"]: s["entry_slot"] - RTH for s in strategies.i2_v1_signals(g, lv)}


def _ok(r):
    return r.has_prev and np.isfinite(r.open_0930) and np.isfinite(r.adr10)


def trend_ext(g, lv, k=0.75, s=0.5, start=30, end=325, exit_idx=359, day="outside", rel="any", tp=None,
              i2map=None, risk_usd=None, tag="TE"):
    """A2-style extension follow with value-area day filters.
    day: all | outside | inside | out_noI2 (I2 has not entered before the signal) | out_I2done (I2 entered before)
    rel: any | toward | away (direction vs value, outside days only)."""
    if i2map is None:
        i2map = i2_entry_idx(g, lv)
    sigs = []
    for r in lv.itertuples():
        if not _ok(r):
            continue
        if day == "inside" and r.i2_dir != 0:
            continue
        if day in ("outside", "out_noI2", "out_I2done") and r.i2_dir == 0:
            continue
        i = r.i
        C = g.C[i, RTH:RTH + 390]
        for idx in range(start + 4, end, 5):
            c = C[idx]
            if not np.isfinite(c):
                continue
            m = c - r.open_0930
            if abs(m) > k * r.adr10:
                if 28 <= idx + 1 <= 32:
                    break
                d = int(np.sign(m))
                had_i2 = i in i2map and i2map[i] <= idx
                if day == "out_noI2" and had_i2:
                    break
                if day == "out_I2done" and not had_i2:
                    break
                if rel == "toward" and d != r.i2_dir:
                    break
                if rel == "away" and d == r.i2_dir:
                    break
                sigs.append({"i": i, "entry_slot": RTH + idx + 1, "dir": d, "sl_pts": s * r.adr10,
                             "tp_pts": (tp * r.adr10 if tp else None), "exit_slot": RTH + exit_idx, "tag": tag})
                break
    return sigs


def at_time(g, lv, T=120, mode="move", day="out_noI2", min_mag=0.0, s=0.5, exit_idx=359, i2map=None,
            need_outside=False, tag="AT"):
    """Single decision at the close of minute T-1, entry at T.
    mode: move (follow open->T move) | away (away from value) | toward (toward value)."""
    if i2map is None:
        i2map = i2_entry_idx(g, lv)
    sigs = []
    for r in lv.itertuples():
        if not _ok(r) or r.i2_dir == 0 and day != "inside" and day != "all":
            continue
        if day == "inside" and r.i2_dir != 0:
            continue
        i = r.i
        if day == "out_noI2" and i in i2map and i2map[i] <= T - 1:
            continue
        c = g.C[i, RTH + T - 1]
        if not np.isfinite(c):
            continue
        m = c - r.open_0930
        if abs(m) < min_mag * r.adr10 or m == 0:
            continue
        if mode == "move":
            d = int(np.sign(m))
        elif mode == "away":
            d = -int(r.i2_dir)
            if np.sign(m) != d:
                continue
        else:
            d = int(r.i2_dir)
            if np.sign(m) != d:
                continue
        if need_outside and r.val <= c <= r.vah:
            continue
        sigs.append({"i": i, "entry_slot": RTH + T, "dir": d, "sl_pts": s * r.adr10, "tp_pts": None,
                     "exit_slot": RTH + exit_idx, "tag": tag})
    return sigs


def inside_accept(g, lv, n_out=3, start=30, last=300, s=0.5, exit_idx=359, stop="adr", tag="ACC"):
    """Inside-open day; after `start`, first time n_out consecutive 5m closes are beyond the same VA edge
    -> follow (acceptance outside value)."""
    sigs = []
    for r in lv.itertuples():
        if not _ok(r) or r.i2_dir != 0:
            continue
        i = r.i
        C = g.C[i, RTH:RTH + 390]
        side, cnt = 0, 0
        for idx in range(4, last, 5):
            c = C[idx]
            if not np.isfinite(c):
                continue
            sd = 1 if c > r.vah else (-1 if c < r.val else 0)
            cnt = cnt + 1 if (sd != 0 and sd == side) else (1 if sd != 0 else 0)
            side = sd
            if cnt >= n_out and idx >= start - 1:
                if 28 <= idx + 1 <= 32:
                    break
                edge = r.vah if sd == 1 else r.val
                sl = s * r.adr10 if stop == "adr" else abs(c - edge) + s * r.adr10 * 0.2
                sigs.append({"i": i, "entry_slot": RTH + idx + 1, "dir": sd, "sl_pts": sl, "tp_pts": None,
                             "exit_slot": RTH + exit_idx, "tag": tag})
                break
    return sigs


def inside_ext_fade(g, lv, k=0.75, s=0.5, start=30, end=325, exit_idx=359, tp="open", need_out_va=False, tag="IXF"):
    """Inside-open day: first 5m close with |close - open| > k*ADR -> FADE it.
    tp: 'open' (back to 09:30 open), 'poc', 'edge' (back to the VA edge it broke), None, or float*ADR."""
    sigs = []
    for r in lv.itertuples():
        if not _ok(r) or r.i2_dir != 0:
            continue
        i = r.i
        C = g.C[i, RTH:RTH + 390]
        for idx in range(start + 4, end, 5):
            c = C[idx]
            if not np.isfinite(c):
                continue
            m = c - r.open_0930
            if abs(m) > k * r.adr10:
                if 28 <= idx + 1 <= 32:
                    break
                if need_out_va and r.val <= c <= r.vah:
                    break
                d = -int(np.sign(m))
                if tp == "open":
                    t = abs(m)
                elif tp == "poc":
                    t = (r.poc - c) * d
                elif tp == "edge":
                    t = ((r.vah if m > 0 else r.val) - c) * d
                elif tp is None:
                    t = None
                else:
                    t = tp * r.adr10
                if t is not None and t < 10:
                    t = 10.0
                sigs.append({"i": i, "entry_slot": RTH + idx + 1, "dir": d, "sl_pts": s * r.adr10, "tp_pts": t,
                             "exit_slot": RTH + exit_idx, "tag": tag})
                break
    return sigs


def pm_migration(g, lv, T=180, ref="vwap", cond="accept", s=0.35, exit_idx=359, day="all", tag="PMM"):
    """Afternoon (entry at idx T, default 12:30) value-migration setup.
    ref='vwap': morning (09:30..T-1) volume-weighted price; accept = morning VWAP beyond prior VAH (VAL) and the
    T-1 close beyond VWAP in the same direction -> follow. cond='reject': morning VWAP beyond the edge but the close
    has come back inside prior value -> trade toward POC (fade the migration)."""
    sigs = []
    for r in lv.itertuples():
        if not _ok(r):
            continue
        if day == "inside" and r.i2_dir != 0:
            continue
        if day == "outside" and r.i2_dir == 0:
            continue
        i = r.i
        a, b = RTH, RTH + T
        tp_ = (g.H[i, a:b] + g.L[i, a:b] + g.C[i, a:b]) / 3
        v = g.V[i, a:b]
        m = np.isfinite(tp_) & np.isfinite(v)
        if m.sum() < 30 or v[m].sum() <= 0:
            continue
        vw = np.sum(tp_[m] * v[m]) / np.sum(v[m])
        c = g.C[i, b - 1]
        if not np.isfinite(c):
            continue
        side = 1 if vw > r.vah else (-1 if vw < r.val else 0)
        if side == 0:
            continue
        if cond == "accept":
            if (c - vw) * side <= 0:
                continue
            d = side
        else:
            if not (r.val <= c <= r.vah):
                continue
            d = -side
        sigs.append({"i": i, "entry_slot": b, "dir": d, "sl_pts": s * r.adr10, "tp_pts": None,
                     "exit_slot": RTH + exit_idx, "tag": tag})
    return sigs


def pm_pullback(g, lv, T=180, s=0.35, exit_idx=359, day="all", tp=None, tag="PMP"):
    """Morning VWAP (09:30..T-1) beyond prior VAH (VAL) = value migrated; T-1 close back INSIDE prior value
    -> trade in the migration direction (buy the pullback into old value)."""
    out = []
    for x in pm_migration(g, lv, T=T, cond="reject", s=s, exit_idx=exit_idx, day=day, tag=tag):
        x = dict(x); x["dir"] = -x["dir"]
        if tp:
            x["tp_pts"] = tp * lv.adr10.iloc[x["i"]]
        out.append(x)
    return out
