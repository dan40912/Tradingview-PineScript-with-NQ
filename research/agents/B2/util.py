"""B2 helpers: load everything once, summary with flat days, bucket tables, permutation test."""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(os.path.dirname(HERE))
for p in (RESEARCH, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from lib import data, features, sim, metrics, strategies  # noqa: E402
import regime  # noqa: E402
import baselines  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)


def load():
    g = data.load_grid("NQ")           # never include_holdout
    lv = features.daily_levels(g)
    rf = regime.regime_features(g, lv)
    ev = regime.event_flags(g.dates)
    return g, lv, rf, ev


def ndays(g, split):
    m = g.tradable & (g.split == split) if split != "ALL" else g.tradable
    return int(m.sum())


def summ(tr, g, split=None, pnl_col="pnl"):
    if split is not None:
        tr = tr[tr.split == split] if split != "ALL" else tr
    nd = ndays(g, split or "ALL")
    if len(tr) == 0:
        return {"n": 0}
    return metrics.summary(tr, n_days=nd, pnl_col=pnl_col)


def table(tr, g, pnl_col="pnl"):
    return pd.DataFrame({s: summ(tr, g, s, pnl_col) for s in ["discovery", "validation", "ALL"]}).T


def attach(tr, rf, ev):
    tr = tr.copy()
    tr = tr.join(rf.drop(columns=["date", "split", "tradable", "has_prev"]), on="i")
    tr = tr.join(ev.drop(columns=["date"]), on="i")
    return tr


def half(tr):
    """discovery sub-periods for stability: D1 = 2024H1, D2 = 2024H2, D3 = 2025H1."""
    d = tr.date.str[:7]
    return np.where(d <= "2024-06", "D1", np.where(d <= "2024-12", "D2", np.where(d <= "2025-06", "D3", "V")))


def bucket_table(tr, col, cuts=None, q=(1 / 3, 2 / 3)):
    """avg pnl per bucket of `col`, cut points from DISCOVERY trades unless given."""
    disc = tr[tr.split == "discovery"]
    if cuts is None:
        ref = disc[col].values.astype(float); ref = ref[np.isfinite(ref)]
        cuts = np.quantile(ref, q)
    b = np.where(np.isfinite(tr[col].values.astype(float)), np.searchsorted(cuts, tr[col].values, side="right"), -1)
    t = tr.assign(b=b, h=half(tr))
    rows = []
    for k in sorted(set(b)):
        if k < 0:
            continue
        x = t[t.b == k]
        row = {"bucket": k}
        for h in ["D1", "D2", "D3", "V"]:
            y = x[x.h == h]
            row[f"{h}_n"] = len(y); row[f"{h}_avg"] = round(y.pnl.mean(), 0) if len(y) else np.nan
        xd = x[x.split == "discovery"]
        row["disc_n"] = len(xd); row["disc_avg"] = round(xd.pnl.mean(), 0) if len(xd) else np.nan
        row["disc_win"] = round(100 * (xd.pnl > 0).mean(), 0) if len(xd) else np.nan
        row["disc_time%"] = round(100 * (xd.reason == "time").mean(), 0) if len(xd) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("bucket"), cuts


def perm_test(tr, keep_mask, g, split="discovery", n=5000, seed=1, stat="avg"):
    """Random removal of the same number of trades (days). Returns p-value for the filter's
    improvement in `stat` (avg / total/maxdd ratio / sharpe)."""
    rng = np.random.default_rng(seed)
    t = tr[tr.split == split].reset_index(drop=True)
    km = np.asarray(keep_mask)[(tr.split == split).values]
    nd = ndays(g, split)

    def f(sub):
        if stat == "avg":
            return sub.pnl.mean()
        s = metrics.summary(sub, n_days=nd)
        return s["sharpe_ann"] if stat == "sharpe" else (s["ret/dd"] if np.isfinite(s["ret/dd"]) else 99)
    obs = f(t[km])
    k = km.sum()
    cnt = 0
    for _ in range(n):
        idx = rng.choice(len(t), k, replace=False)
        if f(t.iloc[np.sort(idx)]) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (n + 1)
