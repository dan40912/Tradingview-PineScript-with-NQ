"""產生「策略進出場地圖」HTML 用的每日資料。

讀 data/NQ_1m_continuous_*.csv.gz，對每個交易日算出：
  - 前一日（同合約）美股盤中 09:30–16:00 的價值區 VAL／POC／VAH（5 點格、70%，和 Pine 指標同一套算法）
  - 前一日盤中高／低／收、隔夜高／低（今天 09:30 前、同一個 trade_date）
  - ATR20（前 20 個交易日盤中振幅平均）
  - 今天 09:30–12:29 的 1 分K（180 根；沒成交的分鐘用前一根收盤補、量 0）
輸出 strategy_map/days_full.json（全部，用來算統計與核對）和 days.json（START 以後，塞進 HTML）。
用法：python3 strategy_map/build_data.py [START]   接著 node strategy_map/build_stats.js、python3 strategy_map/build_html.py

研究用途，不是投資建議。
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "NQ_1m_continuous_20240102-20261019.csv.gz"
QUALITY = ROOT / "data" / "NQ_ES_daily_quality.csv"
HERE = Path(__file__).resolve().parent
START = sys.argv[1] if len(sys.argv) > 1 else "2025-07-01"   # HTML 裡放哪一天以後
BIN = 5.0
VA_FRAC = 0.70


def value_area(bars: pd.DataFrame):
    """和 Pine 指標相同：每根 1 分K 的量平均分到它碰到的每個 5 點格。"""
    prof: dict[int, float] = {}
    for lo, hi, v in zip(bars["low"].values, bars["high"].values, bars["volume"].values):
        a = int(math.floor(lo / BIN))
        b = int(math.floor(hi / BIN))
        nb = b - a + 1
        for k in range(a, b + 1):
            prof[k] = prof.get(k, 0.0) + v / nb
    if not prof:
        return None
    keys = sorted(prof)
    kmin, kmax = keys[0], keys[-1]
    total = sum(prof.values())
    poc, pocv = kmin, -1.0
    for k in keys:                      # 由低往高，同量保留較低的格
        if prof[k] > pocv:
            poc, pocv = k, prof[k]
    lo = hi = poc
    acc = pocv
    while acc < VA_FRAC * total and (lo > kmin or hi < kmax):
        lv = prof.get(lo - 1, -1.0)
        hv = prof.get(hi + 1, -1.0)
        if hv > lv:                     # 同量時先往下擴張
            hi += 1
            acc += max(0.0, hv)
        else:
            lo -= 1
            acc += max(0.0, lv)
    return lo * BIN, poc * BIN, (hi + 1) * BIN


def main():
    df = pd.read_csv(SRC, usecols=["timestamp_utc", "open", "high", "low", "close", "volume", "contract", "trade_date", "is_rth"])
    ts = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert("America/New_York")
    df["mod"] = ts.dt.hour * 60 + ts.dt.minute
    dates = sorted(df["trade_date"].unique())
    groups = {d: g for d, g in df.groupby("trade_date", sort=True)}
    q = pd.read_csv(QUALITY, encoding="utf-8-sig")
    q = q[q["symbol"] == "NQ"].set_index("trade_date")
    # 和凍結回測一致：假日短盤、盤中有缺分鐘的日子不交易
    skip = set(q.index[(q["day_quality"] != "normal") | (q["missing_minutes_rth"] > 0)])
    skip.add("2024-07-25")   # 凍結回測的資料規則排除這天（研究紀錄第 47 步）

    out = []
    prev = None          # 前一日盤中資料
    ranges = []          # 歷來盤中振幅（算 ATR20）
    for d in dates:
        g = groups[d]
        rth = g[(g["mod"] >= 570) & (g["mod"] <= 959)]
        if rth.empty:
            continue
        ct = rth["contract"].iloc[0]
        pre = g[(g["mod"] < 570) | (g["mod"] >= 1080)]          # 前一晚 18:00 起的夜盤到 09:29（不含今天收盤後）
        pre = pre[pre["contract"] == ct]
        morning = rth[rth["mod"] <= 749].set_index("mod")       # 09:30–12:29
        rec = None
        if prev is not None and prev["ct"] == ct and d not in skip and len(morning) >= 150:
            va = prev["va"]
            atr20 = sum(ranges[-20:]) / 20 if len(ranges) >= 20 else None
            o = []; h = []; l = []; c = []; v = []
            last = float(morning["open"].iloc[0])
            for m in range(570, 750):
                if m in morning.index:
                    r = morning.loc[m]
                    o.append(float(r["open"])); h.append(float(r["high"])); l.append(float(r["low"]))
                    c.append(float(r["close"])); v.append(int(r["volume"]))
                    last = float(r["close"])
                else:
                    o.append(last); h.append(last); l.append(last); c.append(last); v.append(0)
            base = round(min(l) * 4)
            tk = lambda arr: [round(x * 4) - base for x in arr]
            rec = {
                "d": d, "ct": ct,
                "val": va[0], "poc": va[1], "vah": va[2],
                "pdh": prev["hi"], "pdl": prev["lo"], "pdc": prev["close"],
                "onh": float(pre["high"].max()) if not pre.empty else None,
                "onl": float(pre["low"].min()) if not pre.empty else None,
                "atr": round(atr20, 2) if atr20 else None,
                "b": base, "o": tk(o), "h": tk(h), "l": tk(l), "c": tk(c), "v": v,
            }
            out.append(rec)
        prev = {"ct": ct, "va": value_area(rth), "hi": float(rth["high"].max()),
                "lo": float(rth["low"].min()), "close": float(rth["close"].iloc[-1])}
        ranges.append(prev["hi"] - prev["lo"])
    (HERE / "days_full.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    recent = [r for r in out if r["d"] >= START]
    (HERE / "days.json").write_text(json.dumps(recent, separators=(",", ":")), encoding="utf-8")
    print(f"全部 {len(out)} 天（{out[0]['d']} ～ {out[-1]['d']}）；HTML 放 {len(recent)} 天（{START} 起）")


if __name__ == "__main__":
    main()
