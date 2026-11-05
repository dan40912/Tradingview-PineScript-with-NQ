# 多代理策略研究協定（2026-11）

研究用途，不是投資建議。

## 目標
在 NQ 日內（1 口，$20/點，每筆成本 $48）找出比 I2 v1 **更賺錢、風險更低**的策略或組合。
「風險更低」以 **最大回撤（maxDD）**、**報酬／回撤（ret/dd）**、**年化日 Sharpe（含沒交易的日子）** 衡量，不是看勝率。

## 共用引擎（一定要用，不要自己寫模擬器）
```python
import sys; sys.path.insert(0, "research")
from lib import data, features, sim, metrics, strategies
g  = data.load_grid("NQ")            # 預設不含 2026（holdout 鎖住）
lv = features.daily_levels(g)        # 前一日價值區、前高低、隔夜高低、09:30 開盤、i2_dir…
sigs = strategies.i2_v1_signals(g, lv)
tr = sim.run_signals(g, sigs)        # 每筆：date, dir, entry, exit, reason, pts, pnl, year, split
print(metrics.by_period(tr, "split")); print(metrics.by_period(tr, "year"))
```
- 網格：每個交易日 1440 格，slot 0 = 前一天 18:00（美東）。09:30 = slot 930（`data.RTH_OFF`），12:29 = 1109，15:59 = 1319。`sim.slot_of("10:15")` 換算。
- 進場＝`entry_slot` 那一分鐘的**開盤價**（訊號必須只用 `entry_slot-1` 以前收盤的資料）。同一分鐘同時碰到止盈和止損，一律算止損。
- `g.tradable[i]` 為 False 的日子（股市休市）不要交易；`lv.has_prev` 為 False（換月或無前一日）表示沒有前一日價值區。
- 引擎已驗證：重現凍結 I2 v1 159/159 筆逐筆一致（`research/verify_engine.py`）。
- ES 資料：`data.load_grid("ES")`（$50/點，成本 $108）。

## 防過度擬合規則（最重要）
| 期間 | 日期 | 用途 |
|---|---|---|
| discovery | 2024-01-02 ～ 2025-06-30 | 自由探索、調參數 |
| validation | 2025-07-01 ～ 2025-12-31 | 只拿**事先選好的少數候選**來確認，不在這裡調參數 |
| holdout | 2026-01-01 ～ 2026-10-19 | **鎖住**。只有最後回測階段（E）可以用 `include_holdout=True` |

1. A／B／C 階段**不准**讀 2026 的資料（不要傳 `include_holdout=True`，也不要讀 backtest/*.csv 裡 2026 的列來調參數）。
2. 記錄你**總共試了幾種變體**（含失敗的），寫在報告裡。
3. 候選門檻（全部要符合）：
   - discovery 與 validation **兩段每筆平均都 > 0**，discovery 每筆 ≥ +$50；
   - discovery 至少 40 筆（或說明為何較少）；
   - 鄰近參數（每個參數往上下各動一格）**多數仍為正**；
   - 和 I2 v1 的重疊：報告在「I2 有交易的日子」與「I2 沒交易的日子」分開的表現。
4. 每個代理最多交出 **3 個候選**，參數完全凍結。

## 交付格式（每個代理）
資料夾 `research/agents/<代號>/`：
- `REPORT.md`（繁體中文、白話）：做了什麼、試了幾種變體、結果表（discovery／validation 分開，含筆數、勝率、每筆、合計、maxDD、ret/dd、Sharpe）、候選與淘汰理由、你的誠實評估。
- `candidates.py`：定義 `CANDIDATES = {"名稱": fn}`，`fn(g, lv) -> list[signal dict]`，可直接丟給 `sim.run_signals(g, fn(g, lv))`。**參數寫死**。函式不得使用未來資料。
- 其他程式放同資料夾。不要改 `research/lib/`（需要新功能就在自己資料夾寫 helper）。不要 git commit。
