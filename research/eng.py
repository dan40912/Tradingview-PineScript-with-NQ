"""陣列版回測引擎：每天一個 dict，RTH 390 分鐘的 O/H/L/C/V（缺的分鐘是 NaN）。"""
import math, pickle, numpy as np, pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
BIN, VA_FRAC, PT, COST = 5.0, 0.70, 20.0, 48.0
FOMC = {"2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18",
        "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-10-29","2025-12-10",
        "2026-01-28","2026-03-18","2026-04-29","2026-06-17","2026-07-29","2026-09-16","2026-10-28","2026-12-09"}

def value_area(lo_, hi_, vol):
    prof = {}
    for l, h, v in zip(lo_, hi_, vol):
        if np.isnan(l): continue
        a, b = int(math.floor(l / BIN)), int(math.floor(h / BIN))
        for k in range(a, b + 1): prof[k] = prof.get(k, 0.0) + v / (b - a + 1)
    keys = sorted(prof); kmin, kmax = keys[0], keys[-1]; total = sum(prof.values()); poc, pv = kmin, -1.0
    for k in keys:
        if prof[k] > pv: pv, poc = prof[k], k
    lo = hi = poc; acc = pv
    while acc < VA_FRAC * total and (lo > kmin or hi < kmax):
        lv = prof.get(lo - 1, -1.0); hv = prof.get(hi + 1, -1.0)
        if hv > lv: hi += 1; acc += max(0.0, hv)
        else: lo -= 1; acc += max(0.0, lv)
    return lo * BIN, (hi + 1) * BIN, poc * BIN

def build(sym="NQ", scale=1.0, extra=None):
    df = pd.read_csv(f"{REPO}/data/{sym}_1m_continuous_20240102-20261019.csv.gz",
                     usecols=["timestamp_utc","open","high","low","close","volume","contract","trade_date"])
    if extra:   # 接在後面的資料（例如 Yahoo 的 10/20 之後），只取主檔最後一天之後的交易日
        e = pd.read_csv(extra, usecols=df.columns.tolist()); df = pd.concat([df, e[e.trade_date > df.trade_date.max()]], ignore_index=True)
    t = pd.to_datetime(df.timestamp_utc).dt.tz_convert("America/New_York")
    df["m"] = (t.dt.hour * 60 + t.dt.minute).values
    for c in ("open","high","low","close"): df[c] *= scale
    q = pd.read_csv(f"{REPO}/data/NQ_ES_daily_quality.csv", encoding="utf-8-sig"); q = q[q.symbol == sym]
    hol = set(q[(q.day_quality.isin(["holiday_short","closed"])) & (q.rth_bars <= 210)].trade_date)
    out, prev, ranges = [], None, []
    for d, g in df.groupby("trade_date", sort=True):
        r = g[(g.m >= 570) & (g.m <= 959)]
        if len(r) == 0: continue
        A = np.full((5, 390), np.nan); idx = (r.m.values - 570)
        for k, c in enumerate(("open","high","low","close","volume")): A[k, idx] = r[c].values
        on = g[(g.m < 570) | (g.m >= 1080)]
        cur = dict(date=d, contract=r.contract.iloc[0], O=A[0], H=A[1], L=A[2], C=A[3], V=A[4],
                   on_hi=on.high.max() if len(on) else np.nan, on_lo=on.low.min() if len(on) else np.nan,
                   pre900=g[g.m == 540].close.iloc[0] if (g.m == 540).any() else np.nan)
        rng = np.nanmax(A[1]) - np.nanmin(A[2])
        if d not in hol and prev is not None and prev["contract"] == cur["contract"]:
            val, vah, poc = value_area(prev["L"], prev["H"], prev["V"])
            tp_ = (A[1] + A[2] + A[3]) / 3; vv = np.nan_to_num(A[4])
            vw = np.cumsum(np.nan_to_num(tp_) * vv) / np.maximum(np.cumsum(vv), 1)
            cur.update(val=val, vah=vah, poc=poc, pdh=np.nanmax(prev["H"]), pdl=np.nanmin(prev["L"]),
                       pclose=prev["C"][~np.isnan(prev["C"])][-1], vwap=vw,
                       pr_rel=(np.nanmax(prev["H"]) - np.nanmin(prev["L"])) / np.median(ranges[-20:]) if len(ranges) >= 20 else np.nan)
            out.append(cur)
        if d not in hol: ranges.append(rng)
        prev = cur
    return out

def sim(D, ei, side, tp, sl, xi, be=None, ha=None, setup=""):
    O, H, L, C = D["O"], D["H"], D["L"], D["C"]
    if ei > 389 or np.isnan(O[ei]): return None
    e = O[ei]; stop = e - side * sl; tgt = e + side * tp; mfe = 0.0; moved = False; last = ei
    for i in range(ei, min(xi, 389) + 1):
        if np.isnan(H[i]): continue
        last = i; h, l = H[i], L[i]
        if (l <= stop) if side == 1 else (h >= stop):
            return dict(setup=setup, side=side, ei=ei, xi=i, res="BE" if moved else "SL", pnl=side * (stop - e) * PT - COST, risk=sl)
        if (h >= tgt) if side == 1 else (l <= tgt):
            return dict(setup=setup, side=side, ei=ei, xi=i, res="TP", pnl=tp * PT - COST, risk=sl)
        mfe = max(mfe, (h - e) if side == 1 else (e - l))
        if be is not None and not moved and mfe >= be * sl: stop = e; moved = True      # 浮盈到 be×R 後，停損移到進場價（下一分鐘起生效）
        if ha and i == ei + ha[0] - 1 and mfe < ha[1] and i + 1 <= 389 and not np.isnan(O[i + 1]):
            return dict(setup=setup, side=side, ei=ei, xi=i + 1, res="HA", pnl=side * (O[i + 1] - e) * PT - COST, risk=sl)
    return dict(setup=setup, side=side, ei=ei, xi=last, res="TIME", pnl=side * (C[last] - e) * PT - COST, risk=sl)
