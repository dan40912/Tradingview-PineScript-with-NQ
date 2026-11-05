"""B2 — reusable daily regime features + US risk-event calendar for NQ research.

Import from repo root:
    import sys; sys.path.insert(0, "research"); sys.path.insert(0, "research/agents/B2")
    from lib import data, features
    import regime
    g = data.load_grid("NQ"); lv = features.daily_levels(g)
    rf = regime.regime_features(g, lv)      # one row per grid day (same index as lv), causal (known by 09:30 ET)
    ev = regime.event_flags(g.dates)        # event calendar flags for any list of 'YYYY-MM-DD' dates

Causality
---------
Every column of regime_features() for day i uses only
  * full sessions of days < i  (daily highs/lows/closes, 1-minute bars), and
  * the overnight session of day i up to 09:29 ET (slots 0..929) plus the 09:30 open
    (the open is the price you would trade at, as in I2's direction rule).
Percentile ranks (`*_pct`) are computed against the PAST `PCT_WIN` days only (no look-ahead).

Event calendar
--------------
Hard-coded public release dates (FOMC statement days, CPI, Employment Situation/NFP), NYSE
holidays, monthly OPEX (3rd Friday; moved to Thursday if the Friday is an NYSE holiday),
quarterly quad witching, CME equity-index roll window (roll Thursday = expiry - 8 calendar days
.. expiry), month start/end, US elections.  Covers 2024-01 .. 2026-12 (dates only).
2026 CPI/NFP dates come from the BLS published schedule as recalled/searched on 2026-11-05; the
Feb-2026 dates were shifted by the Jan/Feb-2026 funding lapse.  Verify before live use.
2025 note: the Oct-2025 shutdown cancelled the Oct-3 NFP and the Nov CPI; Sept CPI came out
Oct-24, Sept NFP Nov-20, Oct+Nov NFP Dec-16, Nov CPI Dec-18.
"""
from __future__ import annotations
import datetime as _dt
import numpy as np
import pandas as pd

PCT_WIN = 250   # trailing window (trading days) for percentile ranks
PCT_MIN = 40    # minimum history before a percentile is defined

# ----------------------------------------------------------------------------------------
# Event calendar (dates only)
# ----------------------------------------------------------------------------------------
FOMC = [  # statement days (14:00 ET)
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]
CPI = [  # 08:30 ET
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-11", "2025-10-24", "2025-12-18",
    "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-11", "2026-10-14", "2026-11-10", "2026-12-10",
]
NFP = [  # Employment Situation, 08:30 ET
    "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05", "2024-05-03", "2024-06-07", "2024-07-05", "2024-08-02",
    "2024-09-06", "2024-10-04", "2024-11-01", "2024-12-06",
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04", "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-11-20", "2025-12-16",
    "2026-01-09", "2026-02-11", "2026-03-06", "2026-04-03", "2026-05-08", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
]
NYSE_HOLIDAYS = [
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04",
    "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
]
NYSE_EARLY_CLOSE = ["2024-07-03", "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24",
                    "2026-11-27", "2026-12-24"]
ELECTIONS = ["2024-11-05", "2026-11-03"]


def _third_fridays(y0=2024, y1=2026):
    hol = set(NYSE_HOLIDAYS)
    out = []
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            d = _dt.date(y, m, 15)
            d = d + _dt.timedelta(days=(4 - d.weekday()) % 7)   # first Friday on/after the 15th = 3rd Friday
            while d.isoformat() in hol:
                d -= _dt.timedelta(days=1)
            out.append(d)
    return out


OPEX = [d.isoformat() for d in _third_fridays()]
QUAD = [d.isoformat() for d in _third_fridays() if d.month in (3, 6, 9, 12)]


def _roll_window():
    s = set()
    for q in _third_fridays():
        if q.month not in (3, 6, 9, 12):
            continue
        # expiry Friday (use the real 3rd Friday even if holiday-shifted)
        start = q - _dt.timedelta(days=8 if q.weekday() == 4 else 7)
        d = start
        while d <= q:
            if d.weekday() < 5:
                s.add(d.isoformat())
            d += _dt.timedelta(days=1)
    return sorted(s)


ROLL_WINDOW = _roll_window()


def _bdays(y0=2023, y1=2027):
    hol = set(NYSE_HOLIDAYS)
    d = pd.bdate_range(f"{y0}-12-01", f"{y1}-01-31")
    return [x.strftime("%Y-%m-%d") for x in d if x.strftime("%Y-%m-%d") not in hol]


def event_flags(dates) -> pd.DataFrame:
    """Boolean event flags for each 'YYYY-MM-DD' date (index = position in `dates`)."""
    dates = [str(d)[:10] for d in dates]
    bd = _bdays()
    pos = {d: k for k, d in enumerate(bd)}
    fomc, cpi, nfp = set(FOMC), set(CPI), set(NFP)
    fomc_next = {bd[pos[d] + 1] for d in FOMC if d in pos}
    fomc_eve = {bd[pos[d] - 1] for d in FOMC if d in pos}
    # month start / end (NYSE trading days)
    ms, me = set(), set()
    for k in range(1, len(bd)):
        if bd[k][:7] != bd[k - 1][:7]:
            ms.add(bd[k]); me.add(bd[k - 1])
    hol = set(NYSE_HOLIDAYS)
    pre_hol, post_hol = set(), set()
    for h in NYSE_HOLIDAYS:
        x = pd.Timestamp(h)
        p = x - pd.offsets.BDay(1)
        while p.strftime("%Y-%m-%d") in hol:
            p -= pd.offsets.BDay(1)
        n = x + pd.offsets.BDay(1)
        while n.strftime("%Y-%m-%d") in hol:
            n += pd.offsets.BDay(1)
        pre_hol.add(p.strftime("%Y-%m-%d")); post_hol.add(n.strftime("%Y-%m-%d"))
    df = pd.DataFrame({"date": dates})
    S = lambda s: df.date.isin(s).values
    df["fomc"] = S(fomc)
    df["fomc_eve"] = S(fomc_eve)
    df["fomc_next"] = S(fomc_next)
    df["cpi"] = S(cpi)
    df["nfp"] = S(nfp)
    df["rel830"] = df.cpi | df.nfp            # 08:30 tier-1 releases
    df["opex"] = S(set(OPEX))
    df["quad"] = S(set(QUAD))
    df["roll_window"] = S(set(ROLL_WINDOW))
    df["month_end"] = S(me)
    df["month_start"] = S(ms)
    df["pre_holiday"] = S(pre_hol) | S(set(NYSE_EARLY_CLOSE))
    df["post_holiday"] = S(post_hol)
    df["election"] = S(set(ELECTIONS))
    wd = pd.to_datetime(df.date).dt.weekday.values
    df["weekday"] = wd
    df["any_tier1"] = df.fomc | df.cpi | df.nfp
    df["any_event"] = df.any_tier1 | df.opex | df.fomc_next | df.month_end | df.election
    return df


# ----------------------------------------------------------------------------------------
# Daily regime features
# ----------------------------------------------------------------------------------------
def _past_pct(x: np.ndarray, win=PCT_WIN, minn=PCT_MIN) -> np.ndarray:
    """Percentile rank (0..1) of x[i] vs x[i-win..i-1] (strictly past). NaN-safe."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if not np.isfinite(x[i]):
            continue
        h = x[max(0, i - win):i]
        h = h[np.isfinite(h)]
        if len(h) >= minn:
            out[i] = (np.sum(h < x[i]) + 0.5 * np.sum(h == x[i])) / len(h)
    return out


def _session_stats(g):
    """Per-day full-session (18:00..17:00) and RTH stats computed from that day's own bars.
    These are only ever used shifted by >= 1 day in regime_features()."""
    n = len(g)
    H, L, C, O = g.H, g.L, g.C, g.O
    out = {k: np.full(n, np.nan) for k in
           ["s_hi", "s_lo", "s_close", "r_hi", "r_lo", "r_open", "r_close", "rv_rth", "park_rth", "gk_rth"]}
    rs = slice(930, 930 + 390)
    for i in range(n):
        h, l, c = H[i], L[i], C[i]
        if np.isfinite(h).any():
            out["s_hi"][i] = np.nanmax(h); out["s_lo"][i] = np.nanmin(l)
            k = np.where(np.isfinite(c))[0]; out["s_close"][i] = c[k[-1]]
        if g.has_rth[i]:
            rh, rl, rc, ro = h[rs], l[rs], c[rs], O[i][rs]
            m = np.isfinite(rc)
            out["r_hi"][i] = np.nanmax(rh); out["r_lo"][i] = np.nanmin(rl)
            k = np.where(m)[0]
            out["r_close"][i] = rc[k[-1]]; out["r_open"][i] = ro[k[0]]
            lc = np.log(rc[m]); r = np.diff(lc)
            out["rv_rth"][i] = np.sqrt(np.sum(r ** 2))                       # realized vol (log units)
            hl = np.log(rh[m] / rl[m])
            out["park_rth"][i] = np.sqrt(np.sum(hl ** 2) / (4 * np.log(2)))  # Parkinson (summed 1m)
            co = np.log(rc[m] / ro[m])
            out["gk_rth"][i] = np.sqrt(max(np.sum(0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2), 0))
    return out


def regime_features(g, lv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Causal pre-09:30 regime features, one row per grid day (index aligned with features.daily_levels)."""
    from lib import features as _f
    if lv is None:
        lv = _f.daily_levels(g)
    n = len(g)
    st = _session_stats(g)
    same = np.array([False] + [g.contract[i] == g.contract[i - 1] for i in range(1, n)])
    # previous real-RTH day index (for RTH-based quantities)
    prev = np.array([g.prev_rth_day(i) if g.prev_rth_day(i) is not None else -1 for i in range(n)])

    # daily true range of FULL session, using prev close only when same contract
    tr = st["s_hi"] - st["s_lo"]
    pc = np.r_[np.nan, st["s_close"][:-1]]
    tr2 = np.where(same, np.maximum.reduce([tr, np.abs(st["s_hi"] - pc), np.abs(st["s_lo"] - pc)]), tr)
    rth_rng = st["r_hi"] - st["r_lo"]
    # same-contract daily log return of RTH close (0 on roll day)
    rc = st["r_close"]
    lr = np.full(n, np.nan)
    for i in range(n):
        j = prev[i]
        if j >= 0 and np.isfinite(rc[i]) and np.isfinite(rc[j]):
            lr[i] = np.log(rc[i] / rc[j]) if g.contract[i] == g.contract[j] else 0.0
    # "known before day i" = values of days < i  -> shift by 1 position (holiday rows kept as-is)
    def lag(a):
        return np.r_[np.nan, a[:-1]]

    s = pd.DataFrame({"date": g.dates})
    TR = pd.Series(lag(tr2))
    for w in (5, 10, 20, 60):
        s[f"atr{w}"] = TR.rolling(w, min_periods=max(3, w // 2)).mean().values
    RR = pd.Series(np.where(g.has_rth, rth_rng, np.nan))
    rr_lag = RR.shift(1)
    s["prev_rth_range"] = (lv.pdh - lv.pdl).values
    s["prev_sess_range"] = lag(tr)
    s["rth_atr10"] = rr_lag.rolling(10, min_periods=5).mean().values
    # realized-vol measures (annualised-ish, % per day) from 1-min RTH bars
    for nm in ["rv_rth", "park_rth", "gk_rth"]:
        x = pd.Series(lag(st[nm]))
        s[f"{nm}10"] = 100 * x.rolling(10, min_periods=5).mean().values
    rv = pd.Series(lag(st["rv_rth"]))
    s["rv5"] = 100 * rv.rolling(5, min_periods=3).mean().values
    s["rv20"] = 100 * rv.rolling(20, min_periods=10).mean().values
    s["rv_ratio"] = s.rv5 / s.rv20                                   # vol expanding (>1) vs contracting
    s["volvol20"] = pd.Series(np.log(lag(tr))).rolling(20, min_periods=10).std().values
    # overnight / opening quantities (day i, up to 09:30)
    on_rng = (lv.onh - lv.onl).values
    s["on_range"] = on_rng
    s["on_range_atr"] = on_rng / s.atr20
    gap = (lv.open_0930 - lv.pdc).values
    s["gap"] = gap
    s["gap_atr"] = gap / s.atr20
    s["absgap_atr"] = np.abs(s.gap_atr)
    s["va_width_atr"] = lv.va_width.values / s.atr20
    s["prev_range_atr"] = s.prev_rth_range / s.atr20
    s["atr5_20"] = s.atr5 / s.atr20
    s["atr20_px"] = 100 * s.atr20 / lv.open_0930.values             # ATR20 as % of price
    # trend (same-contract chained log returns, known before day i)
    LR = pd.Series(lag(lr)).fillna(0.0)
    cum = LR.cumsum()
    for w in (5, 10, 15, 20, 30, 40, 60):
        s[f"ret{w}"] = 100 * (cum - cum.shift(w)).values   # % log-return over the last w sessions (before day i)
    s["trend_eff20"] = (np.abs(cum - cum.shift(20)) / LR.abs().rolling(20).sum()).values   # Kaufman efficiency
    # distance of 09:30 open from 20-day mean of chained price, in ATR units
    lvl = np.exp(cum)  # synthetic chained index (1.0 at start)
    ma20 = pd.Series(lvl).rolling(20, min_periods=10).mean()
    s["dist_ma20_pct"] = 100 * (lvl / ma20 - 1).values
    # drawdown from 60-day high of chained index (as % and in ATR% units)
    hi60 = pd.Series(lvl).rolling(60, min_periods=10).max()
    s["dd60_pct"] = 100 * (lvl / hi60 - 1).values
    s["dd60_atr"] = s.dd60_pct / s.atr20_px
    # downside semivol share over 20d
    neg = LR.clip(upper=0)
    s["down_share20"] = ((neg ** 2).rolling(20, min_periods=10).sum() / (LR ** 2).rolling(20, min_periods=10).sum()).values
    # percentiles vs past 250 days
    for c in ["atr20", "atr5", "rv_rth10", "on_range_atr", "absgap_atr", "volvol20",
              "atr20_px", "dd60_pct", "rv_ratio"]:
        s[c + "_pct"] = _past_pct(s[c].values)
    # NQ "fear" proxy (no VIX): high recent vol + vol expanding + price under recent high + downside-heavy vol
    #   each component is a causal percentile in [0,1]; fear in [0,1]
    s["down_share20_pct"] = _past_pct(s.down_share20.values)
    s["fear"] = (s.rv_rth10_pct + s.rv_ratio_pct + (1 - s.dd60_pct_pct) + s.down_share20_pct) / 4.0
    s["fear_pct"] = _past_pct(s.fear.values)
    s["tradable"] = lv.tradable.values
    s["has_prev"] = lv.has_prev.values
    s["split"] = g.split
    return s


def bucket(x, q=(1 / 3, 2 / 3), ref=None):
    """Assign tercile (default) labels 0..k using quantiles of `ref` (e.g. discovery-only values)."""
    ref = x if ref is None else ref
    ref = np.asarray(ref, float); ref = ref[np.isfinite(ref)]
    cuts = np.quantile(ref, q)
    out = np.where(np.isfinite(x), np.searchsorted(cuts, x, side="right"), -1)
    return out, cuts
