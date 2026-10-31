"""確認回測引擎和凍結規則一致：重算 I2 v1，逐筆對照 backtest/i2_v1_trades.csv。"""
import pickle, pandas as pd, strat
from eng import REPO
ds = pickle.load(open("NQ_days2.pkl", "rb")); ref = pd.read_csv(f"{REPO}/backtest/i2_v1_trades.csv")
mine = {d["date"]: t for d in ds for t in [strat.i2(d)] if t}
hhmm = lambda i: f"{9 + (30 + i) // 60:02d}:{(30 + i) % 60:02d}"
ok = sum(1 for _, r in ref.iterrows() if r.date in mine and abs(mine[r.date]["pnl"] - r.net_usd) < 1 and hhmm(mine[r.date]["ei"]) == r.entry_time_et)
print(f"I2 v1 一致 {ok} / {(ref.date <= '2026-10-19').sum()}（資料到 2026-10-19）")
