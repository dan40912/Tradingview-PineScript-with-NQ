"""B3: reusable pre-decision feature builder for ML / statistical studies.

Every feature for a decision event (day i, entry_slot e) uses ONLY bars that closed at or
before minute e-1 (entry is at the OPEN of minute e, same convention as lib.sim), plus
prior-day / overnight levels from lib.features.daily_levels.

Usage
-----
    import sys; sys.path.insert(0, "research"); sys.path.insert(0, "research/agents/B3")
    from lib import data, features
    import features_ml as fm
    g = data.load_grid("NQ"); lv = features.daily_levels(g)
    ctx = fm.Context(g, lv, es=data.load_grid("ES"))      # precompute indicator tables once
    X = fm.build(ctx, events)    # events: DataFrame with columns i, entry_slot

Notes
-----
* Prices are back-adjusted (by the roll gap seen in the data itself) ONLY for multi-day
  indicator computation, so rolls do not create fake jumps.  Structural distances use raw
  prices of the same contract (lv already blanks roll days via has_prev).
* Classic indicators on 1/5/15/60-minute bars (clock aligned; overnight bars included) and
  on daily RTH bars (only completed previous days).
* CCI uses a normal-approximation mean-absolute-deviation (0.7979 * std) for speed.
* Missing values are left as NaN (GBM handles them; the logistic pipeline imputes with the
  training-window median).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

RTH_OFF = 930
TF_LIST = ["1min", "5min", "15min", "60min"]


# --------------------------------------------------------------------------------------
# minute series (back-adjusted)
# --------------------------------------------------------------------------------------
def _ts_matrix(g):
    base = (pd.to_datetime(g.dates) - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).values
    return base  # datetime64 per day; ts = base[i] + slot minutes


def roll_adjustment(g):
    """Additive adjustment per day so the series is continuous across rolls (latest unadjusted)."""
    n = len(g)
    gaps = np.zeros(n)
    for i in range(1, n):
        if g.contract[i] != g.contract[i - 1]:
            o = g.O[i][np.isfinite(g.O[i])]
            c = g.C[i - 1][np.isfinite(g.C[i - 1])]
            if len(o) and len(c):
                gaps[i] = o[0] - c[-1]
    # day i prices + sum of gaps of later rolls (> i) -> old contracts shifted up to new
    adj = np.zeros(n)
    acc = 0.0
    for i in range(n - 1, -1, -1):
        adj[i] = acc
        acc += gaps[i]
    return adj


def minute_frame(g, adjust=True):
    adj = roll_adjustment(g) if adjust else np.zeros(len(g))
    base = _ts_matrix(g)
    ii, ss = np.where(np.isfinite(g.C))
    ts = base[ii] + ss.astype("timedelta64[m]")
    df = pd.DataFrame({
        "o": g.O[ii, ss] + adj[ii], "h": g.H[ii, ss] + adj[ii], "l": g.L[ii, ss] + adj[ii],
        "c": g.C[ii, ss] + adj[ii], "v": np.nan_to_num(g.V[ii, ss])}, index=pd.DatetimeIndex(ts))
    df = df[~df.index.duplicated()].sort_index()
    return df


def resample(df, tf):
    if tf == "1min":
        return df.copy()
    r = df.resample(tf, label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
    return r.dropna(subset=["c"])


# --------------------------------------------------------------------------------------
# indicators
# --------------------------------------------------------------------------------------
def _wilder(x, n):
    return x.ewm(alpha=1.0 / n, adjust=False).mean()


def indicators(b: pd.DataFrame) -> pd.DataFrame:
    c, h, l = b.c, b.h, b.l
    out = pd.DataFrame(index=b.index)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, 14)
    d = c.diff()
    up, dn = _wilder(d.clip(lower=0), 14), _wilder((-d).clip(lower=0), 14)
    out["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    out["macdh"] = (macd - macd.ewm(span=9, adjust=False).mean()) / atr
    out["macd"] = macd / atr
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    out["bbp"] = (c - (ma - 2 * sd)) / (4 * sd)
    out["bbw"] = 4 * sd / atr
    # ADX
    um, dm = h.diff(), -l.diff()
    pdm = np.where((um > dm) & (um > 0), um, 0.0)
    mdm = np.where((dm > um) & (dm > 0), dm, 0.0)
    pdi = 100 * _wilder(pd.Series(pdm, index=b.index), 14) / atr
    mdi = 100 * _wilder(pd.Series(mdm, index=b.index), 14) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    out["adx"] = _wilder(dx, 14)
    out["dmi"] = (pdi - mdi)
    out["atr_rel"] = atr / atr.rolling(100, min_periods=20).mean()
    e20, e50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    out["ema20_slope"] = (e20 - e20.shift(5)) / atr
    out["ema20_dist"] = (c - e20) / atr
    out["ema50_dist"] = (c - e50) / atr
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    out["stoch"] = (100 * (c - ll) / (hh - ll).replace(0, np.nan)).rolling(3).mean()
    tp = (h + l + c) / 3
    out["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * 0.7979 * tp.rolling(20).std())
    l20, h20 = l.rolling(20).min(), h.rolling(20).max()
    out["donch"] = (c - l20) / (h20 - l20).replace(0, np.nan)
    out["mom10"] = (c - c.shift(10)) / atr
    out["ret1"] = (c - pc) / atr
    out["_atr"] = atr
    return out


IND_KEEP = ["rsi", "macdh", "macd", "bbp", "bbw", "adx", "dmi", "atr_rel", "ema20_slope", "ema20_dist",
            "ema50_dist", "stoch", "cci", "donch", "mom10", "ret1"]


def daily_bars(g, adjust=True):
    """RTH daily bars (09:30-15:59), adjusted, one row per day with has_rth."""
    adj = roll_adjustment(g) if adjust else np.zeros(len(g))
    rO, rH, rL, rC, rV = g.rth(g.O), g.rth(g.H), g.rth(g.L), g.rth(g.C), g.rth(g.V)
    rows = []
    for i in range(len(g)):
        if not g.has_rth[i]:
            continue
        cc = rC[i][np.isfinite(rC[i])]
        oo = rO[i][np.isfinite(rO[i])]
        rows.append({"i": i, "o": oo[0] + adj[i], "h": np.nanmax(rH[i]) + adj[i], "l": np.nanmin(rL[i]) + adj[i],
                     "c": cc[-1] + adj[i], "v": np.nansum(rV[i])})
    return pd.DataFrame(rows).set_index("i")


# --------------------------------------------------------------------------------------
# context (precomputed tables) and feature builder
# --------------------------------------------------------------------------------------
class Context:
    def __init__(self, g, lv, es=None):
        self.g, self.lv, self.es = g, lv.set_index("i", drop=False), es
        self.base = _ts_matrix(g)
        m = minute_frame(g)
        self.tf = {}
        for tf in TF_LIST:
            b = resample(m, tf)
            ind = indicators(b)
            step = pd.Timedelta(tf)
            end = (b.index + step).values  # bar is complete at this timestamp
            self.tf[tf] = (end, ind[IND_KEEP].values.astype(float))
        # daily indicators, shifted: row for day i = indicators of last completed RTH day before i
        db = daily_bars(g)
        dind = indicators(db)
        dind["ret5"] = (db.c - db.c.shift(5)) / dind["_atr"]
        dind["rng_rel"] = (db.h - db.l) / dind["_atr"]
        self.dcols = IND_KEEP + ["ret5", "rng_rel"]
        n = len(g)
        self.daily = np.full((n, len(self.dcols)), np.nan)
        self.atrd = np.full(n, np.nan)
        rth_days = db.index.values
        for i in range(n):
            k = np.searchsorted(rth_days, i) - 1  # last rth day strictly before i
            if k >= 0:
                j = rth_days[k]
                self.daily[i] = dind.loc[j, self.dcols].values
                self.atrd[i] = dind.loc[j, "_atr"]
        # cumulative RTH volume per day (for relative volume)
        rV = np.nan_to_num(g.rth(g.V))
        self.cumv = np.cumsum(rV, axis=1)
        self.rth_days = rth_days
        # ES aligned by date
        if es is not None:
            self.es_idx = np.array([es.dindex.get(d, -1) for d in g.dates])
            self.es_lv = None
            # ES previous RTH close (same contract) per NQ day
            esC = es.rth(es.C)
            self.es_pdc = np.full(n, np.nan)
            for i in range(n):
                k = self.es_idx[i]
                if k < 0:
                    continue
                j = es.prev_rth_day(k)
                if j is not None and es.contract[j] == es.contract[k]:
                    cc = esC[j][np.isfinite(esC[j])]
                    if len(cc):
                        self.es_pdc[i] = cc[-1]


def _last_close(arr, upto):
    """last finite value of arr[: upto+1]"""
    seg = arr[: upto + 1]
    k = np.where(np.isfinite(seg))[0]
    return seg[k[-1]] if len(k) else np.nan


def build(ctx: Context, events: pd.DataFrame) -> pd.DataFrame:
    """events: columns i, entry_slot. Returns feature DataFrame aligned with events (same index)."""
    g, lv = ctx.g, ctx.lv
    I = events["i"].values.astype(int)
    E = events["entry_slot"].values.astype(int)
    ts = ctx.base[I] + E.astype("timedelta64[m]")
    feats = {}
    # multi-timeframe classic indicators
    for tf in TF_LIST:
        end, vals = ctx.tf[tf]
        k = np.searchsorted(end, ts, side="right") - 1
        ok = k >= 0
        tag = tf.replace("min", "m")
        for j, name in enumerate(IND_KEEP):
            col = np.full(len(I), np.nan)
            col[ok] = vals[k[ok], j]
            feats[f"{tag}_{name}"] = col
    for j, name in enumerate(ctx.dcols):
        feats[f"d_{name}"] = ctx.daily[I, j]
    # structural
    rows = {k: np.full(len(I), np.nan) for k in [
        "s_vah", "s_val", "s_poc", "s_vaw", "s_open_poc", "s_i2dir", "s_pdh", "s_pdl", "s_pdc", "s_onh", "s_onl",
        "s_onvwap", "s_gap", "s_onrng", "s_pdrng", "s_since_open", "s_rth_pos", "s_rth_rng", "s_or5_pos", "s_or15_pos",
        "s_or15_rng", "s_vwap", "s_rvol_rth", "s_rvol_on", "s_mins", "s_dow", "s_in_va", "s_open_in_va",
        "s_ret15", "s_ret60m", "x_es_rel_open", "x_es_rel_60", "x_es_rel_on", "x_es_ret_open"]}
    on_vol = lv["on_vol"].reindex(range(len(g))).values
    for n_, (i, e) in enumerate(zip(I, E)):
        r = lv.loc[i]
        atr = ctx.atrd[i]
        C, H, L, V = g.C[i], g.H[i], g.L[i], g.V[i]
        P = _last_close(C, e - 1)
        op = r.open_0930
        rows["s_mins"][n_] = e - RTH_OFF
        rows["s_dow"][n_] = pd.Timestamp(r.date).dayofweek
        if not np.isfinite(atr) or not np.isfinite(P):
            continue
        if r.has_prev:
            vaw = r.vah - r.val
            rows["s_vah"][n_] = (P - r.vah) / atr
            rows["s_val"][n_] = (P - r.val) / atr
            rows["s_poc"][n_] = (P - r.poc) / atr
            rows["s_vaw"][n_] = vaw / atr
            rows["s_open_poc"][n_] = (op - r.poc) / max(vaw, 5)
            rows["s_i2dir"][n_] = r.i2_dir
            rows["s_in_va"][n_] = float(r.val <= P <= r.vah)
            rows["s_open_in_va"][n_] = float(r.val <= op <= r.vah)
            rows["s_pdh"][n_] = (P - r.pdh) / atr
            rows["s_pdl"][n_] = (P - r.pdl) / atr
            rows["s_pdc"][n_] = (P - r.pdc) / atr
            rows["s_gap"][n_] = (op - r.pdc) / atr
            rows["s_pdrng"][n_] = (r.pdh - r.pdl) / atr
        if np.isfinite(r.onh):
            rows["s_onh"][n_] = (P - r.onh) / atr
            rows["s_onl"][n_] = (P - r.onl) / atr
            rows["s_onvwap"][n_] = (P - r.on_vwap) / atr
            rows["s_onrng"][n_] = (r.onh - r.onl) / atr
        # relative overnight volume vs prior 20 days
        prior = [j for j in ctx.rth_days if j < i][-20:]
        if len(prior) >= 5 and np.isfinite(on_vol[i]):
            rows["s_rvol_on"][n_] = on_vol[i] / np.nanmean(on_vol[prior])
        if e > RTH_OFF and np.isfinite(op):
            hh = np.nanmax(H[RTH_OFF:e]); ll_ = np.nanmin(L[RTH_OFF:e])
            rows["s_since_open"][n_] = (P - op) / atr
            rows["s_rth_rng"][n_] = (hh - ll_) / atr
            rows["s_rth_pos"][n_] = (P - ll_) / max(hh - ll_, 1)
            if e >= RTH_OFF + 5:
                h5, l5 = np.nanmax(H[RTH_OFF:RTH_OFF + 5]), np.nanmin(L[RTH_OFF:RTH_OFF + 5])
                rows["s_or5_pos"][n_] = (P - (h5 + l5) / 2) / max(h5 - l5, 1)
            if e >= RTH_OFF + 15:
                h15, l15 = np.nanmax(H[RTH_OFF:RTH_OFF + 15]), np.nanmin(L[RTH_OFF:RTH_OFF + 15])
                rows["s_or15_pos"][n_] = (P - (h15 + l15) / 2) / max(h15 - l15, 1)
                rows["s_or15_rng"][n_] = (h15 - l15) / atr
            tp = (H[RTH_OFF:e] + L[RTH_OFF:e] + C[RTH_OFF:e]) / 3
            vv = V[RTH_OFF:e]
            m = np.isfinite(tp) & np.isfinite(vv)
            if vv[m].sum() > 0:
                rows["s_vwap"][n_] = (P - np.sum(tp[m] * vv[m]) / vv[m].sum()) / atr
            k = e - RTH_OFF - 1
            prior_c = [ctx.cumv[j, k] for j in prior if g.has_rth[j]]
            if len(prior_c) >= 5 and np.mean(prior_c) > 0:
                rows["s_rvol_rth"][n_] = ctx.cumv[i, k] / np.mean(prior_c)
        p15 = _last_close(C, e - 16)
        p60 = _last_close(C, e - 61)
        rows["s_ret15"][n_] = (P - p15) / atr
        rows["s_ret60m"][n_] = (P - p60) / atr
        # NQ vs ES (returns in bp)
        if ctx.es is not None and ctx.es_idx[i] >= 0:
            es = ctx.es; k = ctx.es_idx[i]
            EP = _last_close(es.C[k], e - 1)
            Eop = es.O[k, RTH_OFF]
            if e > RTH_OFF and np.isfinite(Eop) and np.isfinite(op):
                rows["x_es_ret_open"][n_] = 1e4 * (EP / Eop - 1)
                rows["x_es_rel_open"][n_] = 1e4 * ((P / op - 1) - (EP / Eop - 1))
            E60 = _last_close(es.C[k], e - 61)
            if np.isfinite(E60) and np.isfinite(p60):
                rows["x_es_rel_60"][n_] = 1e4 * ((P / p60 - 1) - (EP / E60 - 1))
            if r.has_prev and np.isfinite(ctx.es_pdc[i]) and np.isfinite(Eop):
                rows["x_es_rel_on"][n_] = 1e4 * ((op / r.pdc - 1) - (Eop / ctx.es_pdc[i] - 1))
    feats.update(rows)
    return pd.DataFrame(feats, index=events.index)


# --------------------------------------------------------------------------------------
# event / label helpers
# --------------------------------------------------------------------------------------
def fixed_time_events(g, lv, hhmm_list=("09:45", "10:00", "10:30")):
    from lib import sim
    rows = []
    for r in lv.itertuples():
        if not r.tradable:
            continue
        for t in hhmm_list:
            rows.append({"i": r.i, "date": r.date, "split": r.split, "year": r.year, "t": t, "entry_slot": sim.slot_of(t)})
    return pd.DataFrame(rows)


def forward_return(g, I, E, h):
    """points from OPEN of entry minute to CLOSE of minute e+h-1 (last available)."""
    out = np.full(len(I), np.nan)
    for n_, (i, e) in enumerate(zip(I, E)):
        o = g.O[i, e]
        c = _last_close(g.C[i], min(e + h - 1, RTH_OFF + 389))
        if np.isfinite(o):
            out[n_] = c - o
    return out
