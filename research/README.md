# 回測程式（研究用，不是投資建議）

用 `data/` 的 NQ／ES 1 分K 重現指標邏輯，並測試 `docs/11_策略假設測試_2026-10.md` 裡的所有假設。需要 Python 3、pandas、numpy。

在這個資料夾裡依序執行：

| 步驟 | 指令 | 做什麼 |
|---|---|---|
| 1 | `python3 build_cache.py` | 讀 `data/` 的 1 分K，算出每天的價值區、前日高低、VWAP，存成 `NQ_days2.pkl`、`ES_days2.pkl`（約 15 秒） |
| 2 | `python3 verify.py` | 重算 I2 v1，和 `backtest/i2_v1_trades.csv` 逐筆對照（應該是 159／159） |
| 3 | `python3 grid.py` | 944 種單一策略，分 2024／2025／2026-01～09／2026-10 統計，存成 `grid2.pkl` |
| 4 | `python3 combo.py` | 各家族前幾名組成 401 種組合 |
| 5 | `python3 robust2.py` | 參數高原、ES 交叉驗證、信賴區間 |
| 6 | `python3 final.py` | 文件第 3 節的組合比較表 |
| 7 | `python3 extra.py` | 新假設：波動過濾、FOMC、最晚進場、I2 半口 |

| 檔案 | 內容 |
|---|---|
| `eng.py` | 資料讀取、價值區（和 Pine 版相同算法）、逐分鐘出場模擬（同一分鐘兩邊都碰到算止損） |
| `strat.py` | 所有策略：`i2`、`trend`（順勢突破）、`pullback`、`orb`、`ibfade`、`vwap_pb`、`pdfade` |

有新的 1 分K 資料時，把 `eng.py` 裡的檔名改掉，從步驟 1 重跑即可。
