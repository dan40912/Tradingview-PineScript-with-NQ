"""Tradovate 版指標 vs Python 回測：逐筆比對、計算 Tradovate 圖上 BUY/SELL 的勝率與風險。

用法（在 repo 根目錄）：
  node research/tradovate_port/harness.js tradovate/NQ_順勢多點_Tradovate.js /tmp/nq.csv NQ data/NQ_1m_continuous_20240102-20261019.csv.gz,data/NQ_1m_yahoo_20261020-20261110.csv.gz
  node research/tradovate_port/harness.js tradovate/NQ_順勢多點_Tradovate.js /tmp/es.csv ES data/ES_1m_continuous_20240102-20261019.csv.gz,data/ES_1m_yahoo_20261020-20261110.csv.gz
  python3 research/tradovate_port/compare.py /tmp/nq.csv /tmp/es.csv
"""
import sys
import numpy as np
import pandas as pd

R = "research/"
ROLL = {"2024-03-12", "2024-06-18", "2024-09-17", "2024-12-17", "2025-03-18", "2025-06-17", "2025-09-16",
        "2025-12-16", "2026-03-17", "2026-06-16", "2026-09-15"}   # 換月日：研究裡不交易（連續合約價差）


def load(p):
    t = pd.read_csv(p)
    et = pd.to_datetime(t.entry_utc).dt.tz_convert("America/New_York")
    t["date"] = (et + pd.Timedelta(hours=6)).dt.strftime("%Y-%m-%d")   # CME 交易日（美東 18:00 起）
    t["et"] = pd.to_datetime(t.entry_utc)
    t["xt"] = pd.to_datetime(t.exit_utc)
    return t[~t.date.isin(ROLL)].copy()


nq = load(sys.argv[1])
es = load(sys.argv[2]) if len(sys.argv) > 2 else None

# 1) 第二層：和 research/final_v18/replica_tier2_trades.csv 逐筆比
t2 = pd.read_csv(R + "final_v18/replica_tier2_trades.csv")
for k in ["C1", "TD10"]:
    r = t2[t2.kind == k]
    j = nq[nq.kind == k]
    m = r.merge(j, on="date", how="outer", suffixes=("_ref", "_js"), indicator=True)
    both = m[m._merge == "both"]
    same = np.isclose(both.entry_ref, both.entry_js) & np.isclose(both.pnl_nq / 2, both.pnl)
    print(f"{k}: 研究 {len(r)} 筆 | Tradovate {len(j)} 筆 | 同日 {len(both)} | 進場和損益相同 {same.sum()} | 只在研究 {list(m[m._merge == 'left_only'].date)} | 只在 Tradovate {list(m[m._merge == 'right_only'].date)}")

# 2) 第一層：和最終 520 筆（ref_tier1_book_520.csv）逐筆比
v = pd.read_csv(R + "tradovate_port/ref_tier1_book_520.csv")
KEY = {"I2": "I2|", "前高空": "AM-PD-k1|空", "前低多": "AM-PD-k1|多", "早盤夜低多": "AM-ON-k1|多",
       "亞洲前低多": "ASIA-PD-k1|多", "亞洲VAL多": "ASIA-VA-k1|多"}
for k, vk in KEY.items():
    vv = v[v.key.str.startswith(vk)]
    m = vv.merge(nq[nq.kind == k], on="date", how="left", suffixes=("_v", "_js"))
    same = np.isclose(m.entry_v, m.entry_js.fillna(-1)) & np.isclose(m.pnl_v, m.pnl_js.fillna(-1e9))
    print(f"{k}: 最終 520 筆裡 {len(vv)} 筆 | Tradovate 進場價和損益相同 {same.sum()}")

# 3) ES 圖確認：NQ 灰色觀察點 + ES 圖同名「ES 同步確認」在同一根或更早
if es is not None:
    for k in ["亞洲前低多", "亞洲VAL多", "早盤夜低多"]:
        a = nq[nq.kind == k].merge(es[es.kind == "ES確認" + k][["date", "et"]], on="date", how="left", suffixes=("", "_es"))
        ok = a.et_es.notna() & (a.et_es <= a.et)
        print(f"{k}: NQ 觀察 {len(a)} | ES 同步 {ok.sum()}（勝率 {(a[ok].pnl > 0).mean():.1%}、{a[ok].pnl.sum():+,.0f}）"
              f" | 沒同步 {(~ok).sum()}（勝率 {(a[~ok].pnl > 0).mean():.1%}、{a[~ok].pnl.sum():+,.0f}）")

# 4) Tradovate 圖上彩色 BUY/SELL（第一層）與紙上（第二層）的勝率、風險
P = [("探索 2024-01～2025-06", "2024-01-01", "2025-06-30"), ("驗證 2025-07～12", "2025-07-01", "2025-12-31"),
     ("測試 2026-01～09", "2026-01-01", "2026-09-30"), ("近期 2026-10-01～11-11", "2026-10-01", "2026-11-11"),
     ("全期", "2024-01-01", "2026-11-11")]


def stats(b):
    b = b.sort_values("xt")
    eq = b.pnl.cumsum()
    dd = (eq - eq.cummax().clip(lower=0)).min()
    day = b.groupby("date").pnl.sum()
    wk = b.groupby(pd.to_datetime(b.date).dt.to_period("W")).pnl.sum()
    mo = b.groupby(pd.to_datetime(b.date).dt.to_period("M")).pnl.sum()
    s = (b.pnl < 0).astype(int)
    run = s.groupby((s != s.shift()).cumsum()).sum().max() if len(b) else 0
    return dict(筆數=len(b), 勝率=f"{(b.pnl > 0).mean() * 100:.1f}%", 每筆=round(b.pnl.mean()), 合計=round(b.pnl.sum()),
                最大回撤=round(dd), 最差一天=round(day.min()), 最差一週=round(wk.min()), 最差一月=round(mo.min()),
                虧損月=f"{(mo < 0).sum()}/{len(mo)}", 最多連虧=int(run), 報酬回撤比=round(b.pnl.sum() / -dd, 1) if dd < 0 else None)


books = {"第一層（彩色 BUY/SELL，10 MNQ）": nq[(nq.tier == 1) & (nq.inPlan == 1)],
         "第二層紙上（C1/TD10，5 MNQ）": nq[nq.tier == 2],
         "第一層＋第二層": nq[((nq.tier == 1) & (nq.inPlan == 1)) | (nq.tier == 2)]}
rows = []
for bn, b in books.items():
    for pn, a, z in P:
        rows.append(dict(組合=bn, 期間=pn, **stats(b[(b.date >= a) & (b.date <= z)])))
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
