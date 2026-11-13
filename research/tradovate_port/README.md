# Tradovate 版指標的驗證

- `harness.js`：模擬 Tradovate 呼叫自訂指標的 `map()`，逐根餵 1 分K，輸出指標內部的交易紀錄。
- `compare.py`：和研究逐筆核對（`ref_tier1_book_520.csv`＝docs/24 第一層最終 520 筆；`../final_v18/replica_tier2_trades.csv`＝第二層），並算勝率、風險。
- `compare_out.txt`：用 2024-01-02～2026-11-11 資料（含 11/11）跑出來的結果。
- `tradovate_nq_trades.csv.gz`、`tradovate_es_confirm.csv.gz`：NQ 圖、ES 圖上指標的全部紀錄（`tier` 1＝第一層、2＝第二層紙上、0＝灰色觀察、−1＝ES 同步確認）。

用法見 `compare.py` 開頭。repo 裡的 `data/` 只到 2026-11-10，而且 10/20 之後是 Yahoo 資料，結果會和 `compare_out.txt` 有幾筆小差異。
說明與結果：[`tradovate/README.md`](../../tradovate/README.md)。
