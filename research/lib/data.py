"""Data loading. All times are America/New_York (ET).

Grid layout
-----------
Each trade_date gets a fixed minute grid of 1440 slots starting at 18:00 ET of the
previous calendar day (CME session open).  slot = minutes since 18:00 ET.
  * RTH 09:30 ET  -> slot 930   (RTH_OFF)
  * RTH 15:59 ET  -> slot 1319
  * idx (minutes since 09:30) = slot - 930; I2 uses idx 0..179 (09:30..12:29)
Missing minutes are NaN (no trade that minute).

SPLITS (anti-overfitting protocol)
----------------------------------
  discovery : 2024-01-02 .. 2025-06-30   explore freely here
  validation: 2025-07-01 .. 2025-12-31   confirm a FEW pre-chosen candidates here
  holdout   : 2026-01-01 .. 2026-10-19   LOCKED. Only the final backtest phase (E) may use it.
load_grid() hides the holdout unless include_holdout=True.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "research", "cache")
DISCOVERY_END = "2025-06-30"
VALIDATION_END = "2025-12-31"
HOLDOUT_START = "2026-01-01"
RTH_OFF = 930          # slot of 09:30 ET
NSLOT = 1440
POINT = {"NQ": 20.0, "ES": 50.0}
COST = {"NQ": 48.0, "ES": 108.0}   # round-trip commission+slippage per contract (project convention)


def split_of(date) -> str:
    d = str(date)[:10]
    if d <= DISCOVERY_END:
        return "discovery"
    if d <= VALIDATION_END:
        return "validation"
    return "holdout"


def _load_minutes(symbol: str) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    pq = os.path.join(CACHE, f"{symbol}_1m.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    fn = [f for f in os.listdir(DATA) if f.startswith(f"{symbol}_1m_continuous") and f.endswith(".csv.gz")][0]
    df = pd.read_csv(os.path.join(DATA, fn))
    et = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert("America/New_York")
    td = pd.to_datetime(df["trade_date"])
    # minutes since 18:00 ET of the previous calendar day
    prev = (td - pd.Timedelta(days=1))
    base = prev.dt.tz_localize(None) + pd.Timedelta(hours=18)
    etn = et.dt.tz_localize(None)
    slot = ((etn - base).dt.total_seconds() // 60).astype(int)
    # Monday sessions start Sunday 18:00 -> trade_date-1 is Sunday; fine. Holidays may extend; clip.
    out = pd.DataFrame({
        "trade_date": df["trade_date"].values,
        "slot": slot.values,
        "et": etn.values,
        "open": df["open"].values, "high": df["high"].values,
        "low": df["low"].values, "close": df["close"].values,
        "volume": df["volume"].values.astype(float),
        "contract": df["contract"].values,
        "is_rth": df["is_rth"].values,
        "day_quality": df["day_quality"].values,
    })
    out.to_parquet(pq)
    return out


class Grid:
    """Per-day minute grids. Arrays shape (ndays, 1440)."""

    def __init__(self, symbol, dates, contract, quality, O, H, L, C, V):
        self.symbol = symbol
        self.dates = np.array(dates)          # 'YYYY-MM-DD' strings
        self.contract = np.array(contract)
        self.quality = np.array(quality)
        self.O, self.H, self.L, self.C, self.V = O, H, L, C, V
        self.point = POINT[symbol]
        self.cost = COST[symbol]
        self.year = np.array([d[:4] for d in self.dates])
        self.split = np.array([split_of(d) for d in self.dates])
        self.dindex = {d: i for i, d in enumerate(self.dates)}

    def __len__(self):
        return len(self.dates)

    # convenience slices -------------------------------------------------
    def rth(self, arr):
        """RTH slice 09:30..15:59 (390 cols)."""
        return arr[:, RTH_OFF:RTH_OFF + 390]

    def prev_rth_day(self, i):
        """Index of the previous day with a real RTH session, or None."""
        j = i - 1
        while j >= 0 and not self.has_rth[j]:
            j -= 1
        return j if j >= 0 else None

    def prev_same_contract(self, i):
        j = self.prev_rth_day(i)
        return j is not None and self.contract[j] == self.contract[i]


def load_grid(symbol: str = "NQ", include_holdout: bool = False, drop_exclude: bool = False) -> Grid:
    df = _load_minutes(symbol)
    if drop_exclude:
        df = df[df.day_quality != "exclude"]
    if not include_holdout:
        df = df[df.trade_date < HOLDOUT_START]
    df = df[(df.slot >= 0) & (df.slot < NSLOT)]
    dates = np.sort(df.trade_date.unique())
    di = {d: i for i, d in enumerate(dates)}
    r = df.trade_date.map(di).values
    s = df.slot.values
    shape = (len(dates), NSLOT)
    arrs = []
    for col in ["open", "high", "low", "close", "volume"]:
        a = np.full(shape, np.nan)
        a[r, s] = df[col].values
        arrs.append(a)
    g = df.groupby("trade_date").agg(contract=("contract", "last"), q=("day_quality", "first"), nrth=("is_rth", "sum"))
    G = Grid(symbol, dates, g.loc[dates, "contract"].values, g.loc[dates, "q"].values, *arrs)
    # equity-market holidays (futures trade 09:30-12:59 only, e.g. July 4, Labor Day): not a real RTH session
    rth_c = G.rth(G.C)
    last_rth = np.array([np.where(~np.isnan(r))[0].max() if np.isfinite(r).any() else -1 for r in rth_c])
    G.rth_bars = g.loc[dates, "nrth"].values
    G.equity_holiday = (last_rth <= 209) & (last_rth >= 0) & (G.quality == "holiday_short")
    G.equity_holiday |= (last_rth < 0)
    # Matches the frozen I2 reference: any day with RTH bars (incl. equity holidays and the 2025-11-28 halt day)
    # is used as the *profile source* for the next day, but equity holidays are not tradable.
    G.has_rth = last_rth >= 0
    G.tradable = G.has_rth & ~G.equity_holiday
    return G
