# 順勢過濾回測（2026-11-25）

研究用途，不是投資建議。規則和通過標準先寫在 [00_事先登記.md](00_事先登記.md)，之後才跑結果。

- `trend.py`：算日K趨勢（T1～T4，換月用 `data/NQ_ES_rollover.csv` 回補價差），套到 v18 計畫和 Tradovate 指標。
- `tradovate_signal_port.py`：你貼的「MNQ Aggressive Risk Signal」照原邏輯改寫成 Python（VWAP 改每日重算）。
- v18 的單：分支 `claude/dreamy-lovelace-54nb9c` 的 `research/tradovate_port/harness.js` 跑 `tradovate/NQ_順勢多點_Tradovate.js`，再用 `research/entry_precision/book6.py` 組計畫。
- 資料：`data/*_1m_continuous_20240102-20261019`＋該分支的 `*_1m_yahoo_20261020-20261110`＋`data/*_1m_yahoo_20261111-20261125`。
- 輸出：`v18_full_with_trend.csv`、`trad_1m_with_trend.csv`、`trad_5m_with_trend.csv`（每筆附 T1～T4）。

結論見 [docs/26](../../docs/26_順勢過濾回測.md)。
