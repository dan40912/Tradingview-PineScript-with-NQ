"""Phase D frozen portfolio definitions ("books") for the NQ intraday multi-agent study.

BOOKS = {name: fn(include_holdout: bool) -> DataFrame}. Each returned row is one trade, already sized and net
of costs, with columns: date, sleeve, symbol, key, dir, entry_time, exit_time, reason, pts, contracts, risk,
pnl, split, year.   include_holdout=True is for Phase E ONLY (it is passed straight to data.load_grid).

Account assumed: $50,000.  Base per-trade risk R = $1,000 (2% of the account) = contracts x stop pts x $/pt
(cost not included).  Contracts are floored to 0.1 (= 1 MNQ / 1 MES); micro cost = 1/10 of the full cost
($4.80 per MNQ, $10.80 per MES round trip), exactly as the shared engine scales cost with `contracts`.

Sleeves (all frozen agent candidates, called unchanged; parameters are NOT re-tuned here):
  A3   A3/A3_C3_POC_SL60_NP30     I2 v1 entry, POC target (>=20), SL 60, no-progress 30m/20pt   (I2 family)
  ORBT C1/C1_ORB15_TP110_TS       ORB15 toward value, TP110/SL145, hold 15:59, trend half-size  (I2 family)
  TE   C2/C2_TE_OUTSIDE           trend-extension follow on outside-value days, SL 0.5*ADR10    (diversifier)
  ES   C3/C3_ES_I2_40_15          I2 v1 entry on ES's own value area, TP40/SL15 ES pts, 12:29    (diversifier)
Conflict rules (causal, applied in entry order inside each day):
  A. at most ONE I2-family sleeve per book (they trade the same days in the same direction);
  B. a new trade OPPOSITE to any open position (NQ or ES) is skipped;
  C. total open stop-risk <= 2R ("heat"); a new trade is cut in 0.1 steps to fit, else skipped.

Books (frozen 2026-11 on 2024-01..2025-12 data only; 2026 never loaded):
  D1_CONSERVATIVE : A3 @ R                                   (8 MNQ)
  D2_BALANCED     : A3 @ R  + TE @ R/2 + ES @ R/2           (core + half-risk satellites)
  D3_EQUAL_RISK   : A3 @ R  + TE @ R   + ES @ R
  D4_AGGRESSIVE   : ORBT @ R + TE @ R  + ES @ R
"""

from __future__ import annotations
import os, sys, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(os.path.dirname(HERE))
if RES not in sys.path:
    sys.path.insert(0, RES)
from lib import data, features, sim, strategies  # noqa: E402

_MODS = {}


def load_agent(agent: str):
    """Import research/agents/<agent>/candidates.py under a unique module name (they are all 'candidates')."""
    if agent in _MODS:
        return _MODS[agent]
    d = os.path.join(RES, "agents", agent)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(f"dcand_{agent}", os.path.join(d, "candidates.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _MODS[agent] = m
    return m


def _signals(key, g, lv):
    """key -> list of frozen signals (each candidate's own default sizing information kept in ct0)."""
    if key == "I2v1":
        return strategies.i2_v1_signals(g, lv)
    agent, name = key.split("/")
    fn = load_agent(agent).CANDIDATES[name]
    if agent == "C2":
        return fn(g, lv, sized=False)          # size ourselves from the frozen stop
    return fn(g, lv)


# stop used for sizing when a candidate has no stop (B3: none) -> notional proxy in points
NO_STOP_PROXY = {"B3/B3_NQES_RS_0945_T120": 145.0}


def raw_sleeve(key, g, lv) -> pd.DataFrame:
    sigs = []
    for s in _signals(key, g, lv):
        s = dict(s)
        sl = s.get("sl_pts")
        s["sl0"] = float(sl) if sl is not None and np.isfinite(sl) else NO_STOP_PROXY.get(key, np.nan)
        s["ct0"] = float(s.get("contracts", 1.0))
        s["contracts"] = 1.0
        s.pop("ct", None)
        sigs.append(s)
    tr = sim.run_signals(g, sigs)
    if len(tr) == 0:
        return tr
    tr["pnl1"] = tr["pnl"]
    tr["symbol"] = g.symbol
    tr["point"] = g.point
    tr["cost"] = g.cost
    tr["key"] = key
    return tr.drop(columns=["pnl"])


def size(tr: pd.DataFrame, risk_usd: float, max_ct: float = 3.0) -> pd.DataFrame:
    tr = tr.copy()
    ct = np.floor(risk_usd * tr.ct0 / (tr.sl0 * tr.point) * 10 + 1e-9) / 10
    tr["contracts"] = np.minimum(ct, max_ct).round(1)
    tr["pnl"] = (tr.pts * tr.point - tr.cost) * tr.contracts
    return tr[tr.contracts >= 0.1].reset_index(drop=True)


class Ctx:
    """Grids + levels for NQ and ES, loaded once."""

    def __init__(self, include_holdout: bool = False):
        self.include_holdout = include_holdout
        self.g = {s: data.load_grid(s, include_holdout=include_holdout) for s in ("NQ", "ES")}
        self.lv = {s: features.daily_levels(self.g[s]) for s in ("NQ", "ES")}
        self._raw = {}

    def raw(self, key):
        if key not in self._raw:
            sym = "ES" if key == "C3/C3_ES_I2_40_15" else "NQ"
            self._raw[key] = raw_sleeve(key, self.g[sym], self.lv[sym])
        return self._raw[key]

    def trading_days(self):
        """Days on which NQ or ES is tradable (denominator for daily Sharpe incl. flat days)."""
        d = set(self.g["NQ"].dates[self.g["NQ"].tradable]) | set(self.g["ES"].dates[self.g["ES"].tradable])
        return pd.Index(sorted(d))


# ----------------------------------------------------------------------------- book assembly
def assemble(ctx: Ctx, sleeves: dict, heat_mult: float = 2.0, risk_ref: float | None = None) -> pd.DataFrame:
    """sleeves = {sleeve_label: (key, risk_usd)}.  Conflict rules, applied minute by minute in entry order
    within each day (all causal: only positions already open at the new trade's entry minute are used):
      B. opposite direction to ANY open position (NQ or ES; the two markets move together) -> skip the new trade;
      C. total open stop-risk (contracts x stop pts x point) may not exceed heat_mult x risk_ref
         (risk_ref = largest sleeve risk); a new trade is cut in 0.1-contract steps to fit, or skipped.
    Rule A (at most one I2-family sleeve per book) is enforced by the book definitions themselves."""
    parts = []
    for prio, (lab, (key, r)) in enumerate(sleeves.items()):
        t = size(ctx.raw(key), r)
        t["sleeve"] = lab
        t["prio"] = prio                              # tie-break for same-minute entries: spec order
        parts.append(t)
    tr = pd.concat(parts, ignore_index=True)
    tr["risk"] = tr.contracts * tr.sl0 * tr.point
    cap = heat_mult * (risk_ref if risk_ref is not None else max(r for _, r in sleeves.values()))
    keep = []
    for _, day in tr.sort_values(["date", "entry_slot", "prio"], kind="mergesort").groupby("date", sort=False):
        open_ = []                                   # rows kept so far today
        for idx, row in day.iterrows():
            live = [o for o in open_ if o.exit_slot >= row.entry_slot]
            if any(o.dir != row.dir for o in live):
                continue
            used = sum(o.risk for o in live)
            room = cap - used
            ct = row.contracts
            if row.risk > room + 1e-6:
                unit = row.sl0 * row.point
                ct = np.floor(room / unit * 10 + 1e-9) / 10
                if ct < 0.1:
                    continue
                row = row.copy()
                row["contracts"] = round(ct, 1)
                row["pnl"] = (row.pts * row.point - row.cost) * row.contracts
                row["risk"] = row.contracts * unit
            open_.append(row)
            keep.append(row)
    out = pd.DataFrame(keep).reset_index(drop=True)
    cols = ["date", "sleeve", "symbol", "key", "dir", "entry_time", "exit_time", "reason", "pts", "contracts",
            "risk", "pnl", "split", "year"]
    return out[cols]


def daily_pnl(tr: pd.DataFrame, days) -> pd.Series:
    return tr.groupby("date")["pnl"].sum().reindex(days, fill_value=0.0)


def stats(dv: pd.Series, tr: pd.DataFrame | None = None) -> dict:
    eq = dv.cumsum().values
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    mdd = float((eq - peak).min())
    tot = float(dv.sum())
    m = dv.groupby(dv.index.str[:7]).sum()
    return {"n": int(len(tr)) if tr is not None else np.nan, "total": round(tot), "maxdd": round(mdd),
            "ret/dd": round(tot / -mdd, 2) if mdd < 0 else np.inf,
            "sharpe": round(dv.mean() / dv.std(ddof=1) * np.sqrt(252), 2),
            "worst_day": round(float(dv.min())), "worst_month": round(float(m.min())),
            "neg_months": int((m < 0).sum())}


def half_of(d: str) -> str:
    return d[:4] + ("H1" if d[5:7] <= "06" else "H2")


def boot_maxdd(dv: pd.Series, horizon=252, n=10000, block=5, seed=0) -> np.ndarray:
    """Stationary-ish block bootstrap of daily P&L (blocks of `block` days) -> maxDD over `horizon` days."""
    rng = np.random.default_rng(seed)
    x = dv.values.astype(float)
    L = len(x)
    nb = int(np.ceil(horizon / block))
    starts = rng.integers(0, L - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :horizon]
    paths = np.cumsum(x[idx], axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((n, 1)), paths], axis=1), axis=1)[:, 1:]
    return (paths - peak).min(axis=1)


# ----------------------------------------------------------------------------- frozen books
ACCOUNT = 50_000.0
R = 1_000.0
HEAT = 2.0
A3 = "A3/A3_C3_POC_SL60_NP30"
ORBT = "C1/C1_ORB15_TP110_TS"
TE = "C2/C2_TE_OUTSIDE"
ES = "C3/C3_ES_I2_40_15"

BOOK_SPECS = {
    "D1_CONSERVATIVE": {"A3": (A3, R)},
    "D2_BALANCED": {"A3": (A3, R), "TE": (TE, R / 2), "ES": (ES, R / 2)},
    "D3_EQUAL_RISK": {"A3": (A3, R), "TE": (TE, R), "ES": (ES, R)},
    "D4_AGGRESSIVE": {"ORBT": (ORBT, R), "TE": (TE, R), "ES": (ES, R)},
}

_CTX = {}


def get_ctx(include_holdout: bool = False) -> Ctx:
    k = bool(include_holdout)
    if k not in _CTX:
        _CTX[k] = Ctx(include_holdout=k)
    return _CTX[k]


def run_book(name: str, include_holdout: bool = False, spec: dict | None = None) -> pd.DataFrame:
    spec = spec or BOOK_SPECS[name]
    return assemble(get_ctx(include_holdout), spec, heat_mult=HEAT, risk_ref=R)


def _mk(name):
    def fn(include_holdout: bool = False) -> pd.DataFrame:
        return run_book(name, include_holdout)
    fn.__name__ = name
    return fn


BOOKS = {name: _mk(name) for name in BOOK_SPECS}
