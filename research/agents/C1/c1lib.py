"""C1 helpers: I2-family = value-area regime + entry timing + exit + risk overlay.

Everything is expressed as signal dicts for the shared engine (research/lib/sim.run_signals).
Custom exits (no-progress) are pre-computed as a causal exit_slot (same construction as A3's
`_noprog_exit`): the decision is taken at the CLOSE of minute e+N-1 using only bars e..e+N-1.

Day universe: lv.has_prev & tradable & open outside prior-day RTH value area (lv.i2_dir != 0).
Direction is ALWAYS toward value (= lv.i2_dir).
"""
from __future__ import annotations
import os, sys, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(os.path.dirname(HERE))
if RES not in sys.path:
    sys.path.insert(0, RES)
from lib import data, features, sim, metrics, strategies  # noqa: E402

RTH = data.RTH_OFF
X1229 = RTH + 179
X1559 = RTH + 389
LAST_IDX = 120            # latest entry idx (11:30), same as I2 v1
BLACKOUT = (28, 32)       # entry idx 28..32 (09:58-10:02) -> no trade that day (same as I2 v1)


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec.loader.exec_module(m)
    return m


# B1 indicators (profile / overnight VA) and B2 regime features (ret20) are re-used, not copied.
B1IND = _load_mod("b1_indicators", os.path.join(RES, "agents", "B1", "indicators.py"))
B2REG = _load_mod("b2_regime", os.path.join(RES, "agents", "B2", "regime.py"))


# ----------------------------------------------------------------------------- day features
def day_features(g, lv):
    """Extra causal per-day columns: ret20 (B2), volpoor_rel (B1 extreme-bin rel. volume on the open side),
    onval/onvah (overnight 18:00-09:29 value area, B1 algorithm)."""
    lv = lv.copy()
    rf = B2REG.regime_features(g, lv)
    lv["ret20"] = rf.ret20.values
    vp = np.full(len(lv), np.nan)
    onl = np.full(len(lv), np.nan); onh = np.full(len(lv), np.nan)
    for r in lv.itertuples():
        if not r.has_prev or r.i2_dir == 0:
            continue
        j = g.prev_rth_day(r.i)
        k, pr = B1IND.profile(g.H[j, RTH:RTH + 390], g.L[j, RTH:RTH + 390], g.V[j, RTH:RTH + 390])
        if pr is not None:
            vp[r.i] = (pr[-1] if r.i2_dir == -1 else pr[0]) / pr.mean()
        val, _, vah = B1IND.va_from_profile(*B1IND.profile(g.H[r.i, :RTH], g.L[r.i, :RTH], g.V[r.i, :RTH]))
        onl[r.i], onh[r.i] = val, vah
    lv["volpoor_rel"] = vp
    lv["onval"], lv["onvah"] = onl, onh
    # open outside overnight VA on the same side as the prior-day VA
    lv["dbl"] = ((lv.i2_dir == -1) & (lv.open_0930 > lv.onvah)) | ((lv.i2_dir == 1) & (lv.open_0930 < lv.onval))
    return lv


_CACHE = {}


def load():
    if "nq" not in _CACHE:
        g = data.load_grid("NQ")                     # holdout locked
        assert not (g.split == "holdout").any()
        lv = day_features(g, features.daily_levels(g))
        _CACHE["nq"] = (g, lv)
    return _CACHE["nq"]


def i2_days(lv):
    m = lv.has_prev & lv.tradable & (lv.i2_dir != 0) & np.isfinite(lv.open_0930)
    return lv[m]


# ----------------------------------------------------------------------------- entries
# Each returns {i: entry_slot} for days in the I2 universe (direction = i2_dir).

def _ok_idx(ei):
    return not (BLACKOUT[0] <= ei <= BLACKOUT[1] or ei > LAST_IDX)


def ent_i2(g, lv, m=2, step=5):
    """I2 v1 engine: m consecutive `step`-minute closes inside VA (m=2, step=5 = frozen I2 v1).
    step=1 -> 1-minute closes."""
    out = {}
    for r in i2_days(lv).itertuples():
        C = g.C[r.i]
        cnt = 0
        start = step - 1
        for idx in range(start, LAST_IDX, step):
            c = C[RTH + idx]
            if np.isnan(c):
                continue
            cnt = cnt + 1 if (r.val <= c <= r.vah) else 0
            if cnt >= m:
                ei = idx + 1
                if _ok_idx(ei):
                    out[r.i] = RTH + ei
                break
    return out


def ent_gap(g, lv, idx=1):
    """Gap-fade toward value: enter at the open of 09:30+idx (no further condition)."""
    return {r.i: RTH + idx for r in i2_days(lv).itertuples()}


def ent_orb(g, lv, N=15, first=True):
    """Opening-range breakout TOWARD value. OR = first N minutes. first=True: the first 1-min close outside the
    OR (either side) must be toward value, else no trade (A1 post-hoc lead). first=False: take the first close
    beyond the OR side toward value, ignoring breaks the other way."""
    out = {}
    for r in i2_days(lv).itertuples():
        i, d = r.i, r.i2_dir
        h = np.nanmax(g.H[i, RTH:RTH + N]); l = np.nanmin(g.L[i, RTH:RTH + N])
        if not (np.isfinite(h) and np.isfinite(l)):
            continue
        C = g.C[i, RTH + N: RTH + LAST_IDX]   # closes idx N..LAST_IDX-1 -> entry idx <= LAST_IDX
        tw = np.where((C < l) if d == -1 else (C > h))[0]
        aw = np.where((C > h) if d == -1 else (C < l))[0]
        if not len(tw):
            continue
        k = tw[0]
        if first and len(aw) and aw[0] < k:
            continue
        ei = N + k + 1
        if _ok_idx(ei):
            out[i] = RTH + ei
    return out


def ent_dbl(g, lv, later=False):
    """Double confirmation: I2 day AND open outside the overnight VA on the same side.
    later=False: I2 v1 entry. later=True: later of (I2 entry on prior VA, I2 entry on overnight VA) (B1 lead)."""
    base = ent_i2(g, lv)
    L = lv.set_index("i")
    out = {}
    for i, es in base.items():
        r = L.loc[i]
        if not r.dbl:
            continue
        if later:
            C = g.C[i]; cnt = 0; e2 = None
            for idx in range(4, LAST_IDX, 5):
                c = C[RTH + idx]
                if np.isnan(c):
                    continue
                cnt = cnt + 1 if (r.onval <= c <= r.onvah) else 0
                if cnt >= 2:
                    ei = idx + 1
                    e2 = RTH + ei if _ok_idx(ei) else None
                    break
            if e2 is None:
                continue
            es = max(es, e2)
        out[i] = es
    return out


# ----------------------------------------------------------------------------- exits
def fill(g, i, es, last=X1229):
    e = es
    while e <= last and np.isnan(g.O[i, e]):
        e += 1
    return (e, g.O[i, e]) if e <= last else (None, np.nan)


def noprog_slot(g, i, d, e, px, N, X, base):
    last = e + N - 1
    if last >= base:
        return base
    fav = np.nanmax(g.H[i, e:last + 1] - px) if d == 1 else np.nanmax(px - g.L[i, e:last + 1])
    return last if fav < X else base


def make_signals(g, lv, entries, exit_spec, tag="C1"):
    """exit_spec dict: tp ('poc' | number | None), poc_floor, sl, np=(N, X) or None, exit ('1229'|'1559').
    Returns list of signal dicts. tp='poc': target = max(dir*(poc - fill), poc_floor)."""
    L = lv.set_index("i")
    ex = exit_spec.get("exit", "1229")
    base = {"1229": X1229, "1559": X1559}.get(ex) if ex in ("1229", "1559") else sim.slot_of(ex)
    out = []
    for i, es in sorted(entries.items()):
        r = L.loc[i]
        d = int(r.i2_dir)
        e, px = fill(g, i, es, base)
        if e is None:
            continue
        tp = exit_spec.get("tp")
        if tp == "poc":
            tp = max(d * (r.poc - px), exit_spec.get("poc_floor", 20.0))
            if exit_spec.get("cap") is not None:
                tp = min(tp, exit_spec["cap"])
        elif tp == "vamid":
            tp = max(d * ((r.val + r.vah) / 2 - px), exit_spec.get("poc_floor", 20.0))
        sl = exit_spec.get("sl")
        xs = base
        npx = exit_spec.get("np")
        if npx:
            xs = noprog_slot(g, i, d, e, px, npx[0], npx[1], base)
        out.append({"i": i, "entry_slot": es, "dir": d, "tp_pts": (float(tp) if tp is not None else None),
                    "sl_pts": (float(sl) if sl is not None else None), "exit_slot": xs, "tag": tag})
    return out


# ----------------------------------------------------------------------------- overlays
def overlay(sigs, lv, volpoor=None, trend=False):
    """volpoor: skip day if volpoor_rel >= threshold (B1 frozen 0.25). trend: 0.5 contract when i2_dir is against
    sign of ret20 (B2 frozen)."""
    L = lv.set_index("i")
    out = []
    for s in sigs:
        r = L.loc[s["i"]]
        if volpoor is not None and not (r.volpoor_rel < volpoor):
            continue
        s = dict(s)
        if trend:
            x = r.ret20
            s["contracts"] = 0.5 if (np.isfinite(x) and np.sign(x) == -s["dir"]) else 1.0
        out.append(s)
    return out


# ----------------------------------------------------------------------------- metrics
HALVES = [("2024H1", "2024-01-01", "2024-06-30"), ("2024H2", "2024-07-01", "2024-12-31"),
          ("2025H1", "2025-01-01", "2025-06-30"), ("2025H2", "2025-07-01", "2025-12-31")]


def ndays(g, lo, hi):
    d = g.dates
    return int((g.tradable & (d >= lo) & (d <= hi)).sum())


def longest_losing_streak(p):
    best = cur = 0
    for x in p:
        cur = cur + 1 if x < 0 else 0
        best = max(best, cur)
    return best


def stats(tr, g, lo, hi):
    n_d = ndays(g, lo, hi)
    if tr is None or len(tr) == 0:
        return {"n": 0, "avg": np.nan, "total": 0.0, "maxdd": 0.0, "ret/dd": np.nan, "sharpe": np.nan,
                "win%": np.nan, "worst": np.nan, "lstreak": 0}
    t = tr[(tr.date >= lo) & (tr.date <= hi)].sort_values("date")
    if len(t) == 0:
        return {"n": 0, "avg": np.nan, "total": 0.0, "maxdd": 0.0, "ret/dd": np.nan, "sharpe": np.nan,
                "win%": np.nan, "worst": np.nan, "lstreak": 0}
    s = metrics.summary(t, n_days=n_d)
    return {"n": s["n"], "win%": s["win%"], "avg": s["avg"], "total": s["total"], "maxdd": s["maxdd"],
            "ret/dd": s["ret/dd"], "sharpe": s["sharpe_ann"], "worst": round(t.pnl.min(), 0),
            "lstreak": longest_losing_streak(t.pnl.values)}


DISC = ("2024-01-01", "2025-06-30")
VAL = ("2025-07-01", "2025-12-31")


def full_row(name, tr, g):
    d = {"variant": name}
    for k, v in stats(tr, g, *DISC).items():
        d[f"D_{k}"] = v
    for h, lo, hi in HALVES[:3]:
        s = stats(tr, g, lo, hi)
        d[f"{h}_n"] = s["n"]; d[f"{h}_avg"] = s["avg"]
    d["halves_pos"] = int(sum((d[f"{h}_avg"] or -1) > 0 for h, _, _ in HALVES[:3]))
    return d


def _attach_contracts(tr, sigs):
    if len(tr):
        cmap = {s["i"]: s.get("contracts", 1.0) for s in sigs}
        tr["contracts"] = tr.i.map(cmap).astype(float)
    return tr


def run(g, sigs):
    return _attach_contracts(sim.run_signals(g, sigs), sigs)


def run_through(g, sigs, tick=0.25):
    """Execution stress: a limit target counts as filled only if price trades THROUGH it by one tick.
    Implemented by widening tp by one tick in the engine and then giving the tick back on tp exits
    (fill stays at the original target price)."""
    s2 = [dict(s, tp_pts=(s["tp_pts"] + tick if s.get("tp_pts") is not None else None)) for s in sigs]
    tr = _attach_contracts(sim.run_signals(g, s2), s2)
    if len(tr):
        m = tr.reason == "tp"
        tr.loc[m, "pnl"] -= tick * g.point * tr.loc[m, "contracts"]
        tr.loc[m, "pts"] -= tick
    return tr


def paired_vs(tr, base, n=5000, seed=0):
    """Per-trade paired comparison on days both trade: mean(diff) and bootstrap 95% CI."""
    a = tr.groupby("date").pnl.sum(); b = base.groupby("date").pnl.sum()
    common = a.index.intersection(b.index)
    diff = (a[common] - b[common]).values
    lo, hi = metrics.bootstrap_ci(diff, n=n, seed=seed)
    return {"n_common": len(common), "diff_avg": round(diff.mean(), 1) if len(diff) else np.nan, "ci_lo": lo, "ci_hi": hi}


def daily_diff_ci(tr, base, g, lo, hi, n=5000, seed=0):
    """Paired bootstrap of DAILY pnl difference over all tradable days of the period (flat days = 0)."""
    days = g.dates[g.tradable & (g.dates >= lo) & (g.dates <= hi)]
    a = tr.groupby("date").pnl.sum().reindex(days).fillna(0).values
    b = base.groupby("date").pnl.sum().reindex(days).fillna(0).values
    d = a - b
    l, h = metrics.bootstrap_ci(d, n=n, seed=seed)
    return {"days": len(days), "daily_diff": round(d.mean(), 1), "ci_lo": l, "ci_hi": h}
