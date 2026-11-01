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

## 借用 day-trader-skill 的想法（`docs/12`）

| 指令 | 做什麼 |
|---|---|
| `python3 timemachine_test.py` | 用 trader-timemachine 的算法（前 60 天同時段 5 分K 隨機重組）當「沒有方向性」的對照組，比較實際損益 |
| `python3 sizing_rules_test.py` | 固定每筆風險的口數（trader-budget），以及它的預設風控（reward:risk ≥ 1、不能放空）套在組合 D 上 |
| `python3 scanner_test.py` | 日線趨勢 scanner（均線、動能、RSI、效率比、20 日新高新低）能不能改善組合 D（`docs/13`） |
| `python3 multimarket_test.py` | NQ、ES 同時掃，誰先出訊號就做誰 |
| `python3 scanner_replay.py` | 每次進場當下跑 scanner（日線型態、ORB15、Time Machine、VWAP），對照實際結果；畫出 2026-10 每筆的圖（`docs/14`） |
| `python3 winrate_opt.py` → `winrate_select.py` → `winrate_portfolio.py` | 勝率最佳化：成交量確認、ATR 停損、滾動歷史回放、VWAP、ORB15、TP/SL（`docs/15`） |
| `python3 crash_days.py` | 劇烈跳空日（開盤離價值區 ≥ 1%、H1 跳過、前日範圍外）的歷史走勢和各種做法（`docs/16`） |

## 10/20～10/30（Yahoo，`docs/17`）
`python3 build_cache.py --yahoo` 會把 `data/*_1m_yahoo_20261016-20261030.csv.gz` 接在主檔後面。之後：

| 指令 | 做什麼 |
|---|---|
| `python3 oct_full.py` | 2026-10 全月，每個策略的逐筆和合計 |
| `python3 oct30_variants.py` | 10/30 大跌日，順勢規則的幾個變化：當天、歷史、10 月 |
| `python3 scanner_replay.py` | 10 月每筆進場的圖（會包含 10/20～10/30） |

跑完記得 `python3 build_cache.py` 回到只用主檔，docs/11～16 的數字才會一樣。
