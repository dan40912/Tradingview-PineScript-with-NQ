# -*- coding: utf-8 -*-
"""P2_quant：產生 strategies.json 與 strategies.txt（只寫規格，不回測）。"""
import json, os, itertools

OUT = os.path.dirname(os.path.abspath(__file__))

FOMC = ["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]
CPI = ["2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
       "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
       "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
       "2025-09-11", "2025-10-24", "2025-12-18",
       "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
       "2026-09-11", "2026-10-14", "2026-11-10", "2026-12-10"]
NFP = ["2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05", "2024-05-03", "2024-06-07", "2024-07-05", "2024-08-02",
       "2024-09-06", "2024-10-04", "2024-11-01", "2024-12-06",
       "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04", "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
       "2025-09-05", "2025-11-20", "2025-12-16",
       "2026-01-09", "2026-02-11", "2026-03-06", "2026-04-03", "2026-05-08", "2026-06-05", "2026-07-02", "2026-08-07",
       "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04"]
CAL_NOTE = ("日期清單與 research/agents/B2/regime.py 相同（FOMC＝聲明日 14:00 ET；CPI／NFP＝08:30 ET）。"
            "bls.gov、federalreserve.gov 在這個環境連不上，2026 年 CPI／NFP 來自 BLS 排程的搜尋結果與記憶；"
            "2026-02 的 CPI（02-13）與 NFP（02-11）受 2026 年初短暫停擺影響、最不確定，2026-04-03 NFP 落在耶穌受難日（股市休市），使用前請核對。"
            "2025-10 的 CPI 與 NFP 因停擺取消；9 月 CPI 延到 10-24、9 月 NFP 延到 11-20、10+11 月 NFP 在 12-16、11 月 CPI 在 12-18。"
            "後備方法（不用清單）：08:30 那根 1 分 K（slot 870）的高低差 ≥ 前 20 天同一根的中位數 × 4 → 當天視為數據日；"
            "這個判斷在 08:31 才知道，只能用在 08:31 以後的進場。")

COMMON = """
==========================================================================================
共用定義（所有策略都用；與 lib/ 的網格一致）
==========================================================================================
- slot(hh:mm) = (hh*60+mm − 1080) mod 1440。常用：03:00=540、06:00=720、08:28=868、08:29=869、08:30=870、
  09:30=930、09:31=931、09:45=945、10:00=960、10:30=990、11:30=1050、12:00=1080、12:29=1109、13:00=1140、
  13:55=1195、13:59=1199、14:00=1200、14:44=1244、14:45=1245、15:00=1260、15:29=1289、15:30=1290、15:45=1305、15:59=1319。
- 「在 hh:mm 的價格」＝ slot(hh:mm)−1 那根 1 分 K 的收盤（在 hh:mm 整點已收盤）。進場價＝訊號後下一分鐘的開盤（sim 的 entry_slot 開盤）。
- O930 = g.O[i,930]；PDC/PDH/PDL = 前一個 RTH 日（09:30–15:59）的收／高／低（features.daily_levels 的 pdc/pdh/pdl）。
- ADRn = 前 n 個 RTH 日 (高−低) 的平均，只用今天以前的日子。
- σd20 = 前 20 個 RTH 日「收盤對收盤」報酬（PDC→PDC，%）的標準差。
- VWAP（只在個別策略當方向/追蹤停損用，不是 VWAP 帶）：09:30 起，Σ(典型價×量)/Σ量，典型價=(H+L+C)/3。
- 停損上限：任何一筆的停損 ≤ 145 點（1 口 NQ 最壞 −$2,900−$48 = −$2,948 < $3,000）。
  下單前照 BRIEF 檢查「今日已實現 − 這筆最壞 − 其他持倉最壞 < −$3,000 就不下」。
- 成本：每筆 1 口 NQ 扣 $48；同一分鐘同時碰止盈止損 → 算止損（sim 已如此）。
- 不抱過 15:59（exit_slot ≤ 1319）；夜盤單 08:28 收盤前出場（exit_slot ≤ 868）。
- 換月：前一日與今日不同合約時（g.prev_same_contract(i) 為 False），所有用到 PDC/PDH/PDL/跳空 的策略當天不做。
- 參數：每個策略 ≤ 3 個要探索的參數、每個 2～4 個值；全部 15 個策略合計 208 種變體（見最後的多重檢定說明）。
- 資料切分照 BRIEF：探索 2024-01-02～2025-06-30、驗證 2025-07-01～2025-12-31、測試 2026-01-01～2026-09-30、近期 2026-10-01～2026-11-11。
"""

S = []

# ---------------------------------------------------------------------------------------------
S.append(dict(
    id="Q01", name="GHLZ 盤中動能：首半小時預測末半小時", family="時段季節性／盤中動能",
    priority="中低（文獻強，但本專案 A2 在 NQ 2024–25 描述性檢定 ≈0）",
    entry_window_et="15:30 進場（slot 1290 開盤），15:59 收盤出場",
    definitions={
        "r1": "首半小時報酬（論文定義含隔夜）= C[i,959]/PDC − 1，即前一日 RTH 收盤 → 今天 10:00",
        "r12": "第 12 個半小時報酬 = C[i,1289]/C[i,1259] − 1，即 15:00 → 15:30",
        "σd20": "前 20 日收盤對收盤報酬標準差",
    },
    entry="15:30 整（slot 1290 開盤）進場，方向 = sign(r1)；條件 |r1| ≥ k1×σd20；若 use_r12=True，還要 sign(r12) = sign(r1)（論文發現兩者合用 R² 更高）。r1=0 不做。每天最多 1 筆。",
    stop="固定點數停損 SL ∈ {40, 80}（遠小於 145 上限）。",
    exit="不設止盈；15:59 收盤（exit_slot 1319）時間出場。",
    params_to_explore={"k1": [0.0, 0.5, 1.0], "use_r12": [False, True], "SL_pts": [40, 80]},
    params_fixed={"exit_slot": 1319, "entry_slot": 1290, "contracts": 1},
    rationale="開盤半小時消化隔夜資訊、尾盤半小時是機構再平衡與資訊較慢的交易者進場的時段；期權造市商負 gamma 避險要順著當天方向追單，造成尾盤延續。效果在高波動、高成交量的日子較強。",
    evidence_sources=[
        {"cite": "Gao, Han, Li & Zhou (2018), Market Intraday Momentum, Journal of Financial Economics 129(2):394–414",
         "url": "https://econpapers.repec.org/RePEc:eee:jfinec:v:129:y:2018:i:2:p:394-414",
         "finding": "SPY 1993–2013：首半小時（從前收算）報酬預測末半小時，R² 1.6%；高波動、經濟衰退、重大消息日更強；首半小時＋第 12 個半小時合用更好。"},
        {"cite": "Li, Sakkas & Urquhart (2022), Intraday time series momentum: global evidence and links to market characteristics, Journal of Financial Markets",
         "url": "https://hull-repository.worktribe.com/OutputFile/3016448",
         "finding": "16 個國際指數中 12 個存在首→末半小時動能。"},
        {"cite": "Zhang & Hua (2024), Market Predictability Before the Closing Bell Rings, Risks 12(11):180",
         "url": "https://www.mdpi.com/2227-9091/12/11/180",
         "finding": "隔夜效應與盤中動能這類已知因子隨時間減弱（反面證據）。"},
        {"cite": "本專案 research/agents/A2/REPORT.md（內部）",
         "url": "（本地檔案）",
         "finding": "NQ 2024–25 描述性檢定：開盤半小時→最後半小時 t≈0～1.4，不顯著。"},
    ],
    critiques=[
        "本專案 A2 已發現 NQ 2024–25 這個效應 ≈0；本規格只在 |r1| 大（k1>0）的日子交易，算是『條件版』，仍可能是 0。",
        "末半小時平均波動約 40–80 點，邊際優勢可能小於 $48 成本。",
        "0DTE 擴大後尾盤的 gamma 結構改變（正 gamma 日會變成均值回歸），效果可能反轉；A2 的『15:30 尾盤反轉』在 discovery 很好、validation 翻負，表示這個時段的方向很不穩定。",
        "收盤前 MOC 買賣不平衡公布（15:50）會造成急拉急殺，SL 40 可能常被掃。",
        "與 Q02 高度相關，回測時要當同一家族計算多重檢定。",
    ],
    expected_1111="r1 = 10:00 價格 / 27,835.75 − 1。10:00 價格落在 28,199～28,353 之間（大約 +1.3%～+1.9%），通常 ≥ 1×σd20 → 15:30 做多。結果取決於 15:30 的價格：當天高點 28,697、收 28,484。如果 15:30 已在 28,600 附近，收盤前跌到 28,484 → SL 40/80 被打到，虧 −$848／−$1,648；如果 15:30 還在 28,450 以下 → 小賺。use_r12=True 時，若 15:00→15:30 是下跌，當天就不做。",
    needed_data=["NQ 1 分 K（RTH 與前一日收盤）", "前 20 日收盤（σd20）"],
))

S.append(dict(
    id="Q02", name="Baltussen 全日動能＋大位移（負 gamma 代理）→ 最後 30 分", family="時段季節性／盤中動能",
    priority="中（理論機制最清楚：負 gamma 日才有效；用『大位移日』當代理）",
    entry_window_et="15:30 或 15:45 進場，15:59 出場",
    definitions={
        "ROD": "到進場前的全日報酬 = C[i,E−1]/PDC − 1（E=進場 slot）",
        "σd20": "前 20 日收盤對收盤報酬標準差",
        "負 gamma 代理": "|ROD| ≥ k×σd20（大位移時造市商通常短 gamma；沒有期權資料時用這個代理）",
    },
    entry="E ∈ {15:30, 15:45}。若 |ROD| ≥ k×σd20，於 slot E 開盤順著 sign(ROD) 進場；每天最多 1 筆。",
    stop="固定停損 SL ∈ {50, 100} 點。",
    exit="15:59 收盤時間出場（不設止盈）。",
    params_to_explore={"k": [0.5, 1.0, 1.5], "entry_time": ["15:30", "15:45"], "SL_pts": [50, 100]},
    params_fixed={"exit_slot": 1319, "contracts": 1},
    rationale="選擇權造市商與槓桿 ETF 在價格大幅移動時必須順勢避險（短 gamma 或每日再平衡），避險單集中在收盤前，造成最後 30 分延續；隔天部分回吐。只在大位移日做，等於只挑負 gamma 機率高的日子。",
    evidence_sources=[
        {"cite": "Baltussen, Da, Lammers & Martens (2021), Hedging Demand and Market Intraday Momentum, Journal of Financial Economics 142(1):377–403",
         "url": "https://econpapers.repec.org/RePEc:eee:jfinec:v:142:y:2021:i:1:p:377-403",
         "finding": "60 多種期貨（股、債、商品、匯）1974–2020：前面整天報酬預測最後 30 分；負 gamma 日很強、正 gamma 日不顯著；隔幾天回吐。"},
        {"cite": "Alpha Architect 摘要：Does gamma hedging actually affect stock prices",
         "url": "https://alphaarchitect.com/hot-topic-does-gamma-hedging-actually-affect-stock-prices/",
         "finding": "標普指數期權的 gamma 暴露為負時，盤中動能強很多。"},
        {"cite": "Soebhag, Baltussen & Da (2025), End-of-Day Reversal（工作論文，Quantpedia Awards 2025 第二名）",
         "url": "https://www.eur.nl/en/news/end-day-reversal-pattern-second-place-quantpedia-awards-2025",
         "finding": "個股最後 30 分是反轉（與指數層級的動能相反）；提醒指數動能不能套用到個股。"},
    ],
    critiques=[
        "沒有 gamma（GEX）資料，|ROD| 只是代理；正 gamma 大位移日（例如上漲日被大量 call 賣方吸收）會失效。",
        "本專案 A2 的『15:30 逆著今天方向』在 discovery 每筆 +$121，代表 2024–2025H1 在這個窗口其實偏反轉；本策略在 discovery 很可能是負的，請誠實報告。",
        "樣本：k=1.5 時每年可能只有 30–50 筆。",
        "與 Q01 重疊（都在 15:30 順勢），應合併算檢定次數。",
    ],
    expected_1111="ROD 在 15:30 約為 (P15:30/27,835.75 − 1)，以當天 28,4xx～28,6xx 計約 +2.2%～+2.9%，幾乎必定 ≥ 1.5σd20 → 15:30（或 15:45）做多。收 28,484；若進場價高於 28,534（SL 50）或 28,584（SL 100）就會被停損，否則到 15:59 出場小賺／小虧。這天的尾盤是從高點 28,697 回落，所以對本策略偏不利。",
    needed_data=["NQ 1 分 K", "前 20 日收盤", "（選用）SPX 0DTE gamma 暴露估計；沒有就用代理"],
))

S.append(dict(
    id="Q03", name="Noise Area 時段雜訊帶突破（Zarattini–Aziz–Barbon）", family="盤中趨勢／波動狀態",
    priority="高（文獻＋本專案 A2_TREND_EXT_FOLLOW 三段皆正，這版改成『時段 σ』與追蹤停損，目標是把 A2 的大回撤砍小）",
    entry_window_et="10:00～15:30 每個整點與半點檢查；15:59 前出場",
    definitions={
        "move_d(T)": "第 d 天在檢查時間 T 的『離開盤絕對報酬』= |C[d, slot(T)−1] / O[d,930] − 1|",
        "σ_T": "前 N 個 RTH 日 move_d(T) 的平均（同一時刻、只用過去的日子）",
        "UB_T": "max(O930, PDC) × (1 + m×σ_T)（上緣；用開盤與前收較高者，吸收跳空）",
        "LB_T": "min(O930, PDC) × (1 − m×σ_T)（下緣）",
        "VWAP_T": "09:30 到 T 的 VWAP",
        "檢查時間 T": "10:00, 10:30, 11:00, …, 15:00, 15:30（12 個點）",
        "RVR（狀態）": "RV5/RV60：前 5 日 vs 前 60 日 RTH 高低差平均之比（可選的波動狀態過濾）",
    },
    entry="在檢查時間 T：若 C[slot(T)−1] > UB_T → 於 slot(T) 開盤做多；若 < LB_T → 做空。每天最多 2 筆（第一筆被停損/出場後，下一個檢查點若再越界才可再進，方向可不同）。",
    stop="(a) 硬停損：進場價 ∓ 145 點以內（sim 的 sl_pts＝min(145, |進場價−對側邊界|)）；(b) 追蹤（trail_mode='vwap_band'）：之後每個檢查點 T'，多單若 C[slot(T')−1] < max(UB_T', VWAP_T') → 在 slot(T') 開盤出場；空單若 > min(LB_T', VWAP_T') 出場。trail_mode='hard_only' 只用 (a)。",
    exit="追蹤停損觸發或 15:59 收盤；不設固定止盈（讓趨勢日跑）。",
    params_to_explore={"m": [1.0, 1.5, 2.0], "N_lookback_days": [10, 14, 20], "trail_mode": ["vwap_band", "hard_only"]},
    params_fixed={"check_times": "10:00..15:30 每 30 分", "max_trades_per_day": 2, "hard_stop_cap_pts": 145, "exit_slot": 1319,
                  "regime_filter": "不加（RVR 只在報告中分組呈現：RVR<0.8、0.8–1.2、>1.2）"},
    rationale="每個時刻都有『正常的雜訊幅度』；價格走出這個帶，代表當天有失衡的資訊或資金流（趨勢日），短 gamma 避險、機構分批執行會讓它延續。用 max(開盤, 前收) 當錨，避免跳空日一開盤就被誤判。VWAP 追蹤停損讓虧損單在價格回到『公平價』時就出場。",
    evidence_sources=[
        {"cite": "Zarattini, Aziz & Barbon (2024/2025), Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY), Swiss Finance Institute RP 24-97",
         "url": "https://www.sfi.ch/de/publications/n-24-97-beat-the-market-an-effective-intraday-momentum-strategy-for-s-p500-etf-spy",
         "finding": "SPY 2007–2024：14 日雜訊帶、半小時檢查、VWAP 追蹤停損、收盤出場，淨年化 19.6%、Sharpe 1.33；Quantpedia Awards 2025 第 4 名。"},
        {"cite": "CXO Advisory 摘要",
         "url": "https://www.cxoadvisory.com/momentum-investing/complex-intraday-time-series-momentum-strategy-applied-to-spy/",
         "finding": "對側邊界當動態追蹤停損、整點/半點執行；報告最大回撤約 −25%（帶波動目標槓桿）。"},
        {"cite": "Gao 等 (2018) 與 Baltussen 等 (2021)",
         "url": "https://econpapers.repec.org/RePEc:eee:jfinec:v:142:y:2021:i:1:p:377-403",
         "finding": "盤中動能存在且在大位移/負 gamma 日更強，為本策略提供機制。"},
        {"cite": "本專案 A2_TREND_EXT_FOLLOW（內部）",
         "url": "（research/agents/A2/REPORT.md、FINAL_CONCLUSION.md）",
         "finding": "固定 0.75×ADR10 延伸就順勢、停損 0.5×ADR10、15:29 出場：每筆 discovery +$443／validation +$125／2026 +$206，但 maxDD −$16k～−$17.5k。"},
    ],
    critiques=[
        "低勝率（文獻與 A2 都約 35–50%），連續停損多；盤整日在邊界附近反覆被打。",
        "論文是 SPY＋波動目標槓桿＋最多 4 倍，NQ 1 口固定部位的結果會不同。",
        "半小時才檢查一次追蹤停損，跳動大的分鐘內可能虧超過預期（硬停損 145 兜底）。",
        "與 A2_TREND_EXT_FOLLOW 概念重疊：若它已被採用，本策略的『增量』要用同日比較來證明。",
        "N 太短（10 天）時 σ_T 易受單一大日影響。",
    ],
    expected_1111="錨：UB 用 max(28,352.75, 27,835.75)=28,352.75，LB 用 27,835.75。以 m=1、NQ 在 11:00 的 σ_T 約 0.5%–0.7% 估計，UB(11:00) ≈ 28,495～28,551；早盤跌到 28,199（在帶內，LB 約 27,700 以下，不會做空）。之後漲到 28,697，第一個收在 UB 之上的檢查點（大概 11:00～13:00）做多，進場約 28,52x～28,60x。追蹤停損 max(UB, VWAP)：尾盤從 28,697 回落到 28,484，若跌破 UB（約 28,5xx）就在下一個半點出場，結果大約 −30～−100 點的小虧或打平；hard_only 版則抱到 15:59 以 28,484 出場，同樣小虧/打平。這種『跳空、回測、再創高、尾盤回落』的日子對本策略不友善，但虧損受控。",
    needed_data=["NQ 1 分 K（含量，用於 VWAP）", "前 N 日同時刻的離開盤報酬"],
))

S.append(dict(
    id="Q04", name="開盤跳空分桶回補（ATR 標準化＋第一根 15 分 K 確認）", family="隔夜／日內報酬分解",
    priority="低（本專案 B2 的『非事件日跳空回補』validation 翻負；這版加了大小分桶與確認，仍屬低先驗）",
    entry_window_et="09:31 或 09:45 進場，12:00 前出場",
    definitions={
        "gap": "O930 − PDC（點）",
        "gz": "|gap| / ADR14",
        "15 分確認（confirm='15m'）": "09:30–09:44 這根 15 分 K：跳空向上時收盤 < O930 且最高 < O930 + 0.25×|gap|；跳空向下時相反",
        "填補目標": "target=0.5 → PDC + 0.5×gap；target=1.0 → PDC",
    },
    entry="gz 落在桶內才做。方向 = −sign(gap)（逆跳空）。confirm='none'：09:31 開盤進場；confirm='15m'：條件成立才在 09:45 開盤進場，不成立當天不做。",
    stop="SL = clip(0.25×ADR14, 30, 145) 點。",
    exit="價格碰到填補目標 → 止盈；否則 12:00（exit_slot 1079）收盤出場。",
    params_to_explore={"gz_bucket": ["[0.10,0.30)", "[0.30,0.60)", "[0.60,1.00)"], "target": [0.5, 1.0], "confirm": ["none", "15m"]},
    params_fixed={"exit_slot": 1079, "skip_contract_roll_days": True, "max_trades_per_day": 1},
    rationale="開盤後第一個小時造市商與做市者要消化隔夜累積的庫存，開盤附近的極端價格常被回吐（庫存風險模型）；小跳空多是雜訊、大跳空常是資訊（gap-and-go），所以用波動標準化分桶，並用第一根 15 分 K 是否延伸當確認。",
    evidence_sources=[
        {"cite": "Grant, Wolf & Yu (2005), Intraday price reversals in the US stock index futures market: A 15-year study, Journal of Banking & Finance 29:1311–1327",
         "url": "https://ideas.repec.org/a/eee/jbfina/v29y2005i5p1311-1327.html",
         "finding": "1987–2002 標普期貨：開盤大幅移動後有高度顯著的盤中反轉，向上的大開盤更明顯；扣掉買賣價差成本後顯著性大降。"},
        {"cite": "Yu, Rentzler & Wolf (2005), NASDAQ-100 Index Futures: Intraday Momentum or Reversal?, Journal of Investment Management 3(3)",
         "url": "https://joim.com/article/nasdaq-100-index-futures-intraday-momentum-or-reversal/",
         "finding": "NQ 期貨：隔夜報酬主要伴隨盤中反轉；前一日報酬則動能與反轉都有，且與星期一有關。"},
        {"cite": "Boyarchenko, Larsen & Whelan (2023), The Overnight Drift, Review of Financial Studies 36(9):3502–3547",
         "url": "https://ideas.repec.org/a/oup/rfinst/v36y2023i9p3502-3547..html",
         "finding": "ES 08:30–10:00 時段平均報酬為負（年化約 −3.9%），與造市商庫存與前一日收盤失衡有關。"},
        {"cite": "Edgeful / NexusFi 實務統計",
         "url": "https://www.edgeful.com/blog/posts/half-gaps-vs-full-gaps",
         "finding": "小跳空回補率高、大跳空低；近 6 個月 NQ 全回補率約 51–55%、ES 約 63–64%；半回補率明顯較高（方法論不透明，只當參考）。"},
    ],
    critiques=[
        "本專案 B2_GAPF_NoNews：discovery +$311／validation −$19，已失敗過；I2 本身就是『開在價值區外回補』，重疊很大。",
        "大跳空（gz>1）是趨勢日的典型樣貌（2026-11-11 就是），逆勢會被打。",
        "停損以 ADR 比例計算，在高波動期接近 145 點上限，風報比可能很差（target=0.5 而 gz 小時目標只有幾十點）。",
    ],
    expected_1111="gap = 28,352.75 − 27,835.75 = +517 點；前一日區間 574 點，ADR14 大約 450～550 → gz ≈ 0.95～1.15。若 gz ≥ 1.0 → 超出所有桶，不做（這天是 gap-and-go，躲過是對的）。若 gz 落在 [0.6,1.0)：confirm='none' 會在 09:31 約 28,353 放空，目標 0.5 = 28,094、1.0 = 27,835.75，都沒到（最低 28,199）；停損 0.25×ADR14 ≈ 115～137 點，之後價格漲到 28,697 → 停損，約 −$2,350～−$2,800。confirm='15m'：09:30–09:44 若收在 28,353 以下且沒有創新高，09:45 放空；此後先跌到 28,199（還沒到目標）再大漲 → 一樣停損。",
    needed_data=["NQ 1 分 K", "PDC、ADR14（daily_levels）"],
))

S.append(dict(
    id="Q05", name="早盤賣壓後 10:00 做多（開盤時段季節性反轉）", family="時段季節性",
    priority="中（機制清楚：開盤庫存賣壓；A2 的雙向『10:00 反轉』不穩，這版只做多並用時段 σ）",
    entry_window_et="10:00 進場（slot 960），12:00/13:00/15:59 出場",
    definitions={
        "r30": "C[i,959] / O930 − 1（09:30→10:00）",
        "σ30": "前 20 日 |C[d,959]/O[d,930] − 1| 的平均",
        "早盤低點 L30": "09:30–09:59 的最低價",
        "gap": "O930 − PDC",
    },
    entry="若 r30 ≤ −k×σ30，且（gap_filter='gap_up' 時要求 gap ≥ 0；'any' 不限），10:00（slot 960）開盤做多。只做多。每天最多 1 筆。",
    stop="SL = clip(進場價 − (L30 − 10), 30, 145) 點（停在早盤低點下 10 點，最少 30、最多 145）。",
    exit="exit_rule='2R'：止盈 = 2×SL 點數，否則 15:59 出場；'13:00'：13:00（exit_slot 1139）時間出場；'15:59'：抱到 15:59。",
    params_to_explore={"k": [0.5, 1.0, 1.5], "exit_rule": ["2R", "13:00", "15:59"], "gap_filter": ["any", "gap_up"]},
    params_fixed={"entry_slot": 960, "direction": "long_only", "contracts": 1},
    rationale="Boyarchenko 等發現美股開盤 08:30–10:00 是報酬為負的時段（造市商在開盤出清前一日收盤累積的庫存），之後時段轉正；跳空向上後的早盤回落常是庫存出清而不是新資訊，10:00 後回到原趨勢。只做多是因為 NQ 長期漂移向上、且 Grant 等發現『向上大開盤』後反轉更強（也就是早盤賣壓）。",
    evidence_sources=[
        {"cite": "Boyarchenko, Larsen & Whelan (2023), The Overnight Drift, RFS 36(9)",
         "url": "https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr917.pdf",
         "finding": "ES 1998–2018：08:30–10:00 年化約 −3.9%；與前一日收盤時的委託失衡（庫存風險）高度相關。"},
        {"cite": "Grant, Wolf & Yu (2005), JBF 29",
         "url": "https://ideas.repec.org/a/eee/jbfina/v29y2005i5p1311-1327.html",
         "finding": "開盤大幅移動後盤中反轉，向上大開盤後更明顯。"},
        {"cite": "Heston, Korajczyk & Sadka (2010), Intraday Patterns in the Cross-section of Stock Returns, Journal of Finance 65(4):1369–1407",
         "url": "https://arxiv.org/pdf/1005.3535.pdf",
         "finding": "一小時內的短期反轉來自暫時的流動性失衡與買賣價差反彈。"},
    ],
    critiques=[
        "A2『10:00 反轉』（雙向）加門檻後翻負；A2『過度延伸反向』18 組幾乎全虧——接刀在 NQ 2024–25 一直是虧的。本版只在『早盤回落』而非『整天延伸』時做多，差異需要回測證明。",
        "2024–2026 是多頭，只做多有方向偏誤；請和『同日 10:00 無條件做多』比較（基準）。",
        "重大利空日（如 2025-04 關稅）會連續停損；停損放在早盤低點下，若低點在 10:00 後才出現就會被掃。",
    ],
    expected_1111="開 28,352.75，09:30–10:30 最低 28,199。若 10:00 收在約 28,290 以下（r30 ≤ −0.2%，以 σ30≈0.35%、k=0.5 約 −50～−60 點）→ 10:00 做多，進場約 28,25x～28,29x。停損在 L30−10：若 10:00 前的低點已是 28,199 → 停損 28,189，SL 約 60～100 點，之後漲到 28,697 → 2R 止盈（+120～+200 點）或 15:59 收 28,484（+190～+230 點）。若 10:00 前低點高於 28,209、28,199 是 10:00 後才出現 → 會先被停損（約 −$1,000～−$2,000）。",
    needed_data=["NQ 1 分 K", "前 20 日 09:30→10:00 報酬"],
))

S.append(dict(
    id="Q06", name="FOMC 公布前漂移（pre-FOMC drift）", family="事件日",
    priority="中低（文獻強但近年衰減；樣本只有 23 次）",
    entry_window_et="FOMC 前一天 14:00–15:59（可選）＋ FOMC 當天 09:31/10:30/12:00 → 13:55",
    definitions={
        "FOMC 日": "聲明 14:00 ET 公布的日子，清單：" + ", ".join(FOMC),
        "FOMC 前一日": "FOMC 日的前一個交易日",
    },
    entry="FOMC 當天在 entry_time 開盤做多；include_eve=True 時，FOMC 前一日 14:00（slot 1200）開盤另做一筆多單，15:59 出場。",
    stop="固定 SL ∈ {60, 120} 點。",
    exit="FOMC 當天 13:55（exit_slot 1195）收盤出場，不碰 14:00 公布；前一日那筆 15:59 出場。不設止盈。",
    params_to_explore={"entry_time": ["09:31", "10:30", "12:00"], "include_eve": [False, True], "SL_pts": [60, 120]},
    params_fixed={"fomc_dates": FOMC, "exit_slot_fomc_day": 1195, "exit_slot_eve": 1319},
    rationale="公布前市場風險溢酬的釋放／流動性提供者的需求，造成公布前 24 小時股價漂移向上；不論決策結果。只拿 14:00 前那段，避開公布本身的跳動風險。",
    evidence_sources=[
        {"cite": "Lucca & Moench (2015), The Pre-FOMC Announcement Drift, Journal of Finance 70(1):329–371（NY Fed Staff Report 512）",
         "url": "https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf",
         "finding": "1994–2011 標普期貨：FOMC 前 24 小時（前日 14:00→當日 14:00）平均 +49bp；其他總經公布沒有這種效果。"},
        {"cite": "Kurov, Wolfe & Gilbert (2021), The disappearing pre-FOMC announcement drift, Finance Research Letters",
         "url": "https://www.skidmore.edu/economics/documents/KurovWolfeGilbert-TheDisappearingPre-FOMC-Announce-Drift-200914.pdf",
         "finding": "2015 年後漂移大幅減弱，只剩有記者會的會議（2019 起每次都有）。"},
        {"cite": "本專案 B2（內部）",
         "url": "（research/agents/B2/REPORT.md）",
         "finding": "FOMC 當天上午很安靜：區間約 0.4 ATR（平常 0.59）——小停損可行，但利潤也小。"},
    ],
    critiques=[
        "樣本極小：2024-01～2026-10 共 23 次（探索 12、驗證 4、測試 7）；必須用『隨機非 FOMC 日同時段做多』的排列檢定，否則只是多頭漂移。",
        "文獻顯示 2015 年後已大幅衰減。",
        "FOMC 當天早上若有 CPI/PPI 等數據會干擾。",
    ],
    expected_1111="2026-11-11 不是 FOMC 日（上一次 10-28，下一次 12-09），也不是前一日 → 不交易。",
    needed_data=["NQ 1 分 K", "FOMC 日期清單（已寫入 params_fixed）"],
))

S.append(dict(
    id="Q07", name="FOMC 記者會後反轉", family="事件日",
    priority="中低（文獻有、樣本小）",
    entry_window_et="FOMC 當天 14:45 或 15:00 進場，15:59 出場",
    definitions={
        "Δ": "C[i,1244] − C[i,1199]（14:00 前一刻 → 14:45，聲明＋記者會前 15 分的反應，點）",
        "ADR10": "前 10 日 RTH 平均區間",
    },
    entry="FOMC 日，若 |Δ| ≥ q×ADR10，於 entry_time 開盤逆著 Δ 方向進場（Δ>0 做空、Δ<0 做多）。",
    stop="固定 SL ∈ {80, 145}。",
    exit="15:59 收盤出場（不設止盈）。",
    params_to_explore={"q": [0.15, 0.30], "entry_time": ["14:45", "15:00"], "SL_pts": [80, 145]},
    params_fixed={"fomc_dates": FOMC, "exit_slot": 1319},
    rationale="14:00 聲明後的第一波是演算法與交易壓力（不是完全的新資訊），記者會中主席發言常修正第一波解讀；文獻發現有記者會的會議之後數小時平均反轉約 60bp。",
    evidence_sources=[
        {"cite": "SSRN 4131740（E-mini 標普，FOMC 記者會後的價格反轉；作者與完整內容請見原文）",
         "url": "https://www.scribd.com/document/704531555/SSRN-id4131740",
         "finding": "有記者會的公布之後約 60bp 反轉、可延續數小時到隔天；沒有記者會的不反轉；作者歸因於交易壓力。"},
        {"cite": "CXO Advisory：Stock market return reversal after FOMC announcements",
         "url": "https://cxoadvisory.com/calendar-effects/stock-market-return-reversal-after-fomc-announcements",
         "finding": "公布後的報酬傾向反轉（多日視窗）。"},
        {"cite": "Jyväskylä 大學碩論 (2025) Anticipatory Market Behavior Around FOMC Press Conferences",
         "url": "https://jyx.jyu.fi/handle/123456789/110990",
         "finding": "2011–2025：公布當天異常行為有限，公布後只有微弱反轉（反面/保守證據）。"},
    ],
    critiques=[
        "23 次事件、再加門檻後更少（可能 10–15 筆），統計上幾乎無法證明；只能當『每年 8 次的額外進場點』，口數宜小。",
        "主席講話 14:30–15:15 期間波動極大，SL 80 容易被掃。",
        "論文樣本以 2011–2022 為主，2024–2026 的升降息循環不同。",
    ],
    expected_1111="不是 FOMC 日 → 不交易。",
    needed_data=["NQ 1 分 K", "FOMC 日期清單", "ADR10"],
))

S.append(dict(
    id="Q08", name="CPI／NFP 公布前夜盤漂移（pre-announcement，夜盤進場點）", family="事件日／夜盤",
    priority="中低（提供夜盤額外進場點；證據是工作論文；需注意與 Lucca–Moench『其他公布無漂移』衝突）",
    entry_window_et="夜盤 03:00 或 06:00 進場，08:28 收盤出場（08:30 公布前）",
    definitions={
        "數據日": "CPI 清單：" + ", ".join(CPI) + "；NFP 清單：" + ", ".join(NFP),
        "日曆說明": CAL_NOTE,
    },
    entry="在數據日（依 event_set）於 entry_time（slot 540 或 720）開盤做多。",
    stop="固定 SL ∈ {40, 80} 點。",
    exit="08:28 收盤（exit_slot 868）出場，一定在 08:30 公布前離場；不設止盈。",
    params_to_explore={"entry_time": ["03:00", "06:00"], "SL_pts": [40, 80], "event_set": ["CPI", "NFP", "CPI+NFP"]},
    params_fixed={"exit_slot": 868, "direction": "long", "cpi_dates": CPI, "nfp_dates": NFP},
    rationale="重大公布前投資人承擔不確定性需要補償，且避險部位在公布前被平掉，造成公布前小幅正報酬；這是夜盤唯一有文獻支持的時段型進場點，和 RTH 策略幾乎不相關。",
    evidence_sources=[
        {"cite": "上海高級金融學院（SAIF, Jun Pan 網頁）工作論文 PreAnn（2021-07），pre-announcement returns",
         "url": "https://en.saif.sjtu.edu.cn/junpan/PreAnn_20210728.pdf",
         "finding": "除 FOMC 外，非農等重要總經公布前也有顯著的公布前報酬；NFP 平均約 10.1bp。"},
        {"cite": "Lucca & Moench (2015) JF",
         "url": "https://libertystreeteconomics.newyorkfed.org/2012/07/the-puzzling-pre-fomc-announcement-drift",
         "finding": "反面：他們發現其他總經公布沒有公布前報酬（樣本與方法不同）。"},
        {"cite": "Boyarchenko 等 (2023) / Liberty Street Economics (2026-07) The Disappearing Overnight Drift",
         "url": "https://libertystreeteconomics.newyorkfed.org/2026/07/the-disappearing-overnight-drift/",
         "finding": "歐洲開盤附近的隔夜漂移在 2021–25 已消失 → 夜盤時段效應會衰減。"},
    ],
    critiques=[
        "每年 CPI+NFP 約 24 天，探索期約 36 筆；10bp ≈ NQ 20～28 點，扣 $48 成本後很薄。",
        "數據日期必須核對（尤其 2025-10/11 停擺、2026-02）。",
        "夜盤流動性較差，滑價可能大於 $48 假設。",
    ],
    expected_1111="11-11 沒有 CPI／NFP（CPI 在 11-10、NFP 在 11-06）→ 不交易。（若是 11-10 當天：03:00 做多、08:28 出場。）",
    needed_data=["NQ 1 分 K（夜盤）", "CPI／NFP 日期清單"],
))

S.append(dict(
    id="Q09", name="CPI／NFP 數據日：08:30 衝擊後的 RTH 延續或反向", family="事件日",
    priority="低（B2 的 08:30 開盤區間突破 validation 大虧；文獻顯示價格在一小時內就消化完）",
    entry_window_et="數據日 09:31 或 10:00 進場，12:29 出場",
    definitions={
        "shock": "C[i,929] − C[i,869]（08:30 公布前一刻 → 09:30 前一刻，點）",
        "sz": "shock / ADR10",
        "數據日": "同 Q08 清單；或用後備偵測（slot 870 區間 ≥ 前 20 日中位數 × 4）",
    },
    entry="數據日，若 |sz| ≥ th，於 entry_time 開盤進場：mode='cont' 順著 shock，mode='fade' 逆著 shock。每天最多 1 筆。",
    stop="SL = min(145, 0.3×ADR10)。",
    exit="12:29 收盤（exit_slot 1109）時間出場，不設止盈。",
    params_to_explore={"mode": ["cont", "fade"], "th": [0.15, 0.30], "entry_time": ["09:31", "10:00"]},
    params_fixed={"exit_slot": 1109, "cpi_dates": CPI, "nfp_dates": NFP, "calendar_note": CAL_NOTE},
    rationale="總經消息在幾秒到一小時內反映；到 09:30 時，現貨開盤帶進新一批參與者（股票帳戶、選擇權避險），可能延續（資訊被確認）或反向（夜盤過度反應）。兩種模式都測，看哪一邊有穩定的符號。",
    evidence_sources=[
        {"cite": "Andersen, Bollerslev, Diebold & Vega (2003, AER; 2007, JIE) 即時價格發現系列",
         "url": "https://www.newyorkfed.org/medialibrary/media/research/current_issues/ci14-6.html",
         "finding": "標普期貨對數據的主要反應在幾秒內完成；NFP 是少數能引起大幅移動的公布（NY Fed Current Issues 14-6）。"},
        {"cite": "LSEG Tick History：How markets react to CPI surprises",
         "url": "https://www.lseg.com/content/dam/data-analytics/en_us/documents/ebooks/lseg-tick-history-research-paper-how-markets-react-to-cpi-surprises.pdf",
         "finding": "CPI 高於預期時標普期貨平均跌約 0.31%；意外越大反應越大。"},
        {"cite": "本專案 B2（內部）",
         "url": "（research/agents/B2/REPORT.md）",
         "finding": "CPI/NFP 日 08:30 開盤區間突破：discovery +$1,033/筆、validation 10 筆 −$1,276/筆 → 淘汰；NFP 上午波動 0.8–1.25 ATR。"},
    ],
    critiques=[
        "樣本小（每年約 22 天、再加門檻），8 種變體裡挑一個很容易是運氣。",
        "B2 已在相近窗口失敗過；預期這個策略大概率 ≈0，主要價值是確認『數據日要不要避開其他策略』。",
        "NFP 2026-04-03 落在股市休市日（耶穌受難日），期貨提早收盤，要排除。",
    ],
    expected_1111="11-11 不是數據日 → 不交易。",
    needed_data=["NQ 1 分 K（含 08:29、09:29）", "CPI／NFP 日期清單", "ADR10"],
))

S.append(dict(
    id="Q10", name="月底月初資金流（Turn-of-the-Month）日內做多", family="日曆效應",
    priority="中低（長期文獻強；期貨上 1990 後有衰減報告；本專案描述性 t<3）",
    entry_window_et="TOM 日 09:31 或 10:00 進場，15:59 出場",
    definitions={
        "T−1": "每月最後一個交易日（用 NYSE 交易日曆；這是日曆資訊，不是價格，不算偷看）",
        "T+1..T+3": "每月第 1～3 個交易日",
        "day_set": "'TOM4' = {T−1, T+1, T+2, T+3}；'FIRST3' = {T+1, T+2, T+3}；'LAST1' = {T−1}",
    },
    entry="day_set 內的日子：entry='09:31' → 09:31 開盤做多；entry='10:00_dip' → 只在 C[i,959] < O930（早盤下跌）時於 10:00 開盤做多，否則不做。",
    stop="固定 SL ∈ {75, 145}。",
    exit="15:59 收盤時間出場（不設止盈）。",
    params_to_explore={"day_set": ["TOM4", "FIRST3", "LAST1"], "entry": ["09:31", "10:00_dip"], "SL_pts": [75, 145]},
    params_fixed={"exit_slot": 1319, "direction": "long", "holiday_calendar": "research/agents/B2/regime.py 的 NYSE_HOLIDAYS"},
    rationale="退休金、薪資、基金申購在月初湧入股市（買盤），月底之前機構為了付款『搶現金』賣股；月底到月初的資金流造成可預期的價格壓力與回補。",
    evidence_sources=[
        {"cite": "McConnell & Xu (2008), Equity Returns at the Turn of the Month, Financial Analysts Journal 64(2)",
         "url": "https://quantpedia.com/strategies/turn-of-the-month-in-equity-indexes",
         "finding": "1926–2005 美股：月底最後一天＋月初三天賺走了全部超額報酬；35 國中 31 國存在。"},
        {"cite": "Etula, Rinne, Suominen & Vaittinen (2020), Dash for Cash: Monthly Market Impact of Institutional Liquidity Needs, RFS 33(1):75–111",
         "url": "https://ideas.repec.org/a/oup/rfinst/v33y2020i1p75-111..html",
         "finding": "月底付款需求造成全球流動市場的系統性價格型態（月底前賣壓、之後回補）。"},
        {"cite": "Maberly & Waggoner (2000), Atlanta Fed WP 2000-11",
         "url": "https://econpapers.repec.org/RePEc:fip:fedawp:2000-11",
         "finding": "反面：標普期貨的 TOM 效應 1990 年後消失。"},
    ],
    critiques=[
        "TOM 的報酬大部分可能在隔夜（收→開），RTH 只拿得到一部分。",
        "本專案 A2 已描述性看過月初月底：每組 70–85 天，t<3。",
        "多頭期間任何『做多』都賺，必須和『隨機 4 天/月做多』基準比較。",
    ],
    expected_1111="11-11 是 11 月第 8 個交易日（11-02 起算）→ 不在任何 day_set → 不交易。",
    needed_data=["NQ 1 分 K", "NYSE 交易日曆"],
))

S.append(dict(
    id="Q11", name="前日收在低點附近（IBS）＋連跌 → 隔天日內做多", family="日線均值回歸",
    priority="中（IBS 在 QQQ 是很有名的效應；但大部分可能在隔夜）",
    entry_window_et="09:31 或 10:00 進場，15:59 出場",
    definitions={
        "IBS": "(PDC − PDL) / (PDH − PDL)，前一 RTH 日收盤在當天區間的位置（0＝收最低、1＝收最高）",
        "streak": "截至前一日，RTH 收盤連續低於前一個 RTH 收盤的天數",
        "ADR10": "前 10 日 RTH 平均區間",
    },
    entry="若 IBS ≤ θ 且 streak ≥ s：entry='09:31' → 09:31 開盤做多；entry='10:00' → 10:00 開盤做多（不加額外條件）。只做多，每天最多 1 筆。",
    stop="SL = min(145, 0.4×ADR10)。",
    exit="15:59 收盤時間出場（不設止盈）。",
    params_to_explore={"theta": [0.15, 0.25, 0.35], "s": [0, 2], "entry": ["09:31", "10:00"]},
    params_fixed={"exit_slot": 1319, "direction": "long"},
    rationale="收在當天低點附近代表收盤前被動賣壓/去槓桿，隔天流動性恢復時回補；短期超賣的指數 ETF（特別是 QQQ）有可靠的一日回歸。",
    evidence_sources=[
        {"cite": "Pagonidis (2013), The IBS Effect: Mean Reversion in Equity ETFs（NAAIM Wagner Award）",
         "url": "https://quantifiedstrategies.substack.com/p/the-internal-bar-strength-ibs-indicator",
         "finding": "美國與多國股票指數 ETF：IBS 低的隔天報酬顯著較高、IBS 高的較低。"},
        {"cite": "Connors & Alvarez (2009), Short Term Trading Strategies That Work；Quantified Strategies 的 SPY RSI+IBS 測試",
         "url": "https://quantifiedstrategies.substack.com/p/mean-reversion-strategy-in-s-and",
         "finding": "IBS<0.25 搭配短期 RSI 的 SPY 做多規則長期正報酬（實務回測，未扣嚴格成本）。"},
        {"cite": "CXO Advisory：連跌之後的報酬",
         "url": "https://www.cxoadvisory.com/?p=3371",
         "finding": "1990–2016：連跌後的反轉比連漲後強，但相對交易成本很小（保守證據）。"},
    ],
    critiques=[
        "文獻多用『收盤進、隔天收盤出』，包含隔夜；我們只能 RTH 進出，效果可能被砍掉一大半。",
        "空頭/崩跌期（如 2025-03～04）會連續虧；停損 0.4×ADR10 在高波動期接近 145。",
        "與 Q05（早盤賣壓做多）可能在同一天同向，需算合併風險。",
    ],
    expected_1111="前一日 H 28,090、L 27,516、C 27,835.75 → IBS = 319.75/574 = 0.557 > 0.35 → 不交易。",
    needed_data=["NQ 1 分 K（前一日 RTH 高低收）", "連續收盤序列"],
))

S.append(dict(
    id="Q12", name="波動壓縮後（NR7／Inside Day）前日高低突破", family="波動狀態",
    priority="低（主要是實務經驗，缺嚴謹回測）",
    entry_window_et="09:35 到 11:30 或 14:00 之間第一次突破",
    definitions={
        "NR7": "前一日 RTH 區間 (PDH−PDL) 是最近 7 個 RTH 日中最小",
        "ID（Inside Day）": "PDH ≤ 前前日高 且 PDL ≥ 前前日低",
        "cond": "'NR7'、'ID'、'NR7_or_ID'",
        "mid": "(PDH + PDL)/2",
    },
    entry="前一日符合 cond，且 O930 落在 [PDL, PDH] 內（開在區間外就不做，避免和跳空/I2 重疊）。09:35 起到 window_end，第一根 1 分 K 收盤 > PDH + 2 → 下一分鐘做多；< PDL − 2 → 做空。每天最多 1 筆。",
    stop="SL = min(145, |進場價 − mid|)。",
    exit="TP = tp_R × SL（tp_R='none' 時不設）；否則 15:59 收盤出場。",
    params_to_explore={"cond": ["NR7", "ID", "NR7_or_ID"], "tp_R": [1.0, 2.0, "none"], "window_end": ["11:30", "14:00"]},
    params_fixed={"start": "09:35", "buffer_pts": 2, "exit_slot": 1319},
    rationale="區間收縮代表多空暫時平衡、選擇權賣方壓低波動；平衡被打破時常出現區間擴張（Crabel 的波動循環）。只在開在區間內的日子做，讓突破是當天才發生的資訊。",
    evidence_sources=[
        {"cite": "Crabel (1990), Day Trading with Short Term Price Patterns and Opening Range Breakout",
         "url": "https://www.luxalgo.com/library/concept/nr4-nr7-narrow-range-bars.md",
         "finding": "NR4/NR7 的定義來源；觀察以期貨為主，但屬交易經驗、沒有發表統計。"},
        {"cite": "Quantifiable Edges (2008) NDX 寬幅日後接 NR7",
         "url": "https://quantifiableedges.blogspot.com/2008/04/what-happens-when-range-rapidly.html",
         "finding": "NDX 1986 起的小型研究（部落格等級）。"},
        {"cite": "Connors & Raschke (1995), Street Smarts（ID/NR4 型態）",
         "url": "https://www.chartmill.com/documentation/trading-and-investing/publications/24-Tsunami-Setups",
         "finding": "實務型態書；無樣本外檢驗。"},
    ],
    critiques=[
        "證據弱；波動聚集（GARCH）意味著低波動後通常仍是低波動，突破常失敗。",
        "trend.py 已有『前日高低突破』，本策略只是加上壓縮條件；筆數會很少（NR7 約 1/7 的日子，再要求開在區內）。",
        "ID 的定義常與 NR7 同時成立，兩個條件的結果高度相關。",
    ],
    expected_1111="前一日區間 574 點（27,516～28,090），通常不是 NR7；且 O930 = 28,352.75 已在 PDH 之上（開在區間外）→ 不交易。",
    needed_data=["NQ 1 分 K", "前 7 日 RTH 高低"],
))

S.append(dict(
    id="Q13", name="NQ 相對 ES 強弱（時段標準化 z 分數）", family="跨市場／相對強弱",
    priority="高（本專案 B3 的乾淨線索三段皆正，但無停損、回撤巨大；這版加 z 標準化與停損）",
    entry_window_et="09:45、10:30 或 13:00 決策；持有 120 分鐘",
    definitions={
        "β60": "前 60 個 RTH 日，NQ 開→收報酬對 ES 開→收報酬的 OLS 斜率（只用過去）",
        "rel_T": "ln(NQ_T/NQ_{T−60}) − β60×ln(ES_T/ES_{T−60})，T−60＝決策前 60 分鐘（用 1 分 K 收盤；09:45 決策時 T−60 落在 08:45 盤前，可接受）",
        "z_T": "rel_T / std(前 20 日同一決策時刻的 rel)",
    },
    entry="在決策時刻 T：mode='mom' → z_T ≥ z0 做多 NQ、z_T ≤ −z0 做空 NQ；mode='rev' → 方向相反。於 slot(T) 開盤進場，每個決策時刻最多 1 筆、每天最多 1 筆。",
    stop="SL = min(145, 0.35×ADR10)（B3 原版沒有停損）。",
    exit="進場後 120 分鐘（exit_slot = entry_slot + 119，且 ≤ 1319）收盤出場；不設止盈。",
    params_to_explore={"mode": ["mom", "rev"], "z0": [1.0, 1.5, 2.0], "decision_time": ["09:45", "10:30", "13:00"]},
    params_fixed={"lookback_min": 60, "hold_min": 120, "beta_window_days": 60, "z_window_days": 20},
    rationale="科技股領頭時（NQ 相對 ES 強），資金輪動與指數期貨套利/ETF 申購會讓強勢延續一段時間（lead-lag 與產業動能）；反向模式則是價差均值回歸。文獻對兩者都有支持，所以都測，並和內部線索比對。",
    evidence_sources=[
        {"cite": "本專案 B3_NQES_RS_0945_T120（FINAL_CONCLUSION.md）",
         "url": "（本地檔案）",
         "finding": "09:45 依 NQ 相對 ES 近 60 分強弱順勢、抱 120 分、無停損：每筆 +$280／+$87／+$314（探索／驗證／2026），和 I2 相關 0.14，但 maxDD −$33,680。"},
        {"cite": "Hou (2007), Industry Information Diffusion and the Lead-lag Effect in Stock Returns, RFS 20(4)；Chordia & Swaminathan (2000), JF 55(2)",
         "url": "https://ideas.repec.org/a/oup/rfinst/v20y2007i4p1113-1138.html",
         "finding": "資訊在不同資產/產業間擴散有延遲，領先者的報酬預測落後者。"},
        {"cite": "Gatev, Goetzmann & Rouwenhorst (2006), Pairs Trading, RFS 19(3)",
         "url": "https://ideas.repec.org/a/oup/rfinst/v19y2006i3p797-827.html",
         "finding": "價差偏離後的均值回歸（日頻、個股）；作為 'rev' 模式的依據。"},
        {"cite": "NexusFi：NQ 與 ES 背離的實務說明",
         "url": "https://nexusfi.com/a/strategies/nq-futures-trading-strategies",
         "finding": "領先的指數傾向延續、落後者是回歸候選（實務經驗，無檢驗）。"},
    ],
    critiques=[
        "需要 NQ 與 ES 分鐘對齊；10/20 以後是 Yahoo 資料，時間戳可能有 1 分鐘偏差，影響 z。",
        "B3 原版的獲利可能主要來自『無停損抱 120 分』；加停損可能讓結果變差，請同時報告無停損版（只當對照、不算進參數）。",
        "科技股特有事件（大型財報）會讓價差持續偏離，rev 模式虧大。",
    ],
    expected_1111="需要 ES 資料才能算。推論：09:45 時 NQ 剛從 28,352 回落、若 ES 跌得較少 → z<0 → 'mom' 模式在 09:45 做空 NQ（之後先跌到 28,199 → 小賺，但 120 分內反彈 → 可能打平或停損）；10:30 時若 NQ 反彈領先 ES → z>0 → 'mom' 做多，抱到 12:30，參與上漲段（若 12:30 前漲向 28,5xx～28,6xx 則賺 +200～+300 點）。實際請以 ES 資料計算。",
    needed_data=["NQ 與 ES 1 分 K（時間對齊）", "前 60 日 RTH 開收（β）", "前 20 日同時刻 rel"],
))

S.append(dict(
    id="Q14", name="時段相對成交量（RVOL）爆量 K 棒：延續或反轉", family="成交量",
    priority="中（文獻對『高量→延續、低量→反轉』有支持；也能增加每天的進場點）",
    entry_window_et="09:35～15:00，每根 5 分 K 收盤時判斷",
    definitions={
        "5 分 K": "RTH 從 09:30 起每 5 分鐘一根（09:30–09:34 為第 0 根），由 1 分 K 聚合",
        "RVOL_b": "V_b / 前 20 日同一根 5 分 K 的成交量中位數",
        "zret_b": "(C_b − O_b) / 前 20 日同一根 |C−O| 的平均",
    },
    entry="09:35 起到 15:00 止，若 RVOL_b ≥ R 且 |zret_b| ≥ 2：mode='follow' 順著該根方向、mode='fade' 逆向，在下一分鐘開盤進場。持倉中不重複進場；每天最多 3 筆。",
    stop="SL = clip(該 5 分 K 的高低差, 20, 145) 點（多單停在該根低點附近、空單停在高點附近的等效點數）。",
    exit="exit='1R'／'2R'：止盈 = 1×／2× SL；'30m'：30 分鐘後收盤出場（不設止盈）。所有單 15:59 前出場。",
    params_to_explore={"R": [2.0, 3.0, 4.0], "mode": ["follow", "fade"], "exit": ["1R", "2R", "30m"]},
    params_fixed={"zret_min": 2.0, "start": "09:35", "end": "15:00", "max_trades_per_day": 3, "vol_lookback_days": 20},
    rationale="異常大量伴隨的價格移動通常是資訊型交易（延續），一般量的急動多是流動性衝擊（反轉）；用同時段的量作基準，避免開盤/收盤的季節性量干擾。",
    evidence_sources=[
        {"cite": "Connolly & Stivers (2003), Momentum and Reversals in Equity-Index Returns During Periods of Abnormal Turnover and Return Dispersion, Journal of Finance 58(4):1521–1556",
         "url": "https://ideas.repec.org/a/bla/jfinan/v58y2003i4p1521-1556.html",
         "finding": "後一週成交量異常高時指數（含指數期貨）報酬延續、異常低時反轉。"},
        {"cite": "Llorente, Michaely, Saar & Wang (2002), Dynamic Volume-Return Relation of Individual Stocks, RFS 15(4)",
         "url": "https://ideas.repec.org/a/oup/rfinst/v15y2002i4p1005-1047.html",
         "finding": "資訊型交易造成的量 → 報酬延續；避險型交易造成的量 → 反轉。"},
        {"cite": "Campbell, Grossman & Wang (1993), Trading Volume and Serial Correlation in Stock Returns, QJE 108(4)",
         "url": "https://ideas.repec.org/a/oup/qjecon/v108y1993i4p905-939..html",
         "finding": "指數層級：高量日的報酬傾向反轉（流動性需求）——與上面方向相反，所以兩個 mode 都測。"},
    ],
    critiques=[
        "10/20 之後是 Yahoo 1 分 K，成交量口徑可能和之前不同，近期段的 RVOL 會失真。",
        "換月週成交量在兩個合約間移轉，RVOL 會異常；換月週應排除或用兩合約合計。",
        "每天最多 3 筆，要嚴格遵守每日 $3,000 風險檢查。",
    ],
    expected_1111="需要 5 分 K 與量才能確定。可能的訊號：(1) 09:35～10:30 下殺到 28,199 時的爆量長黑 → 'follow' 放空（之後反彈，1R 可能先到也可能停損）、'fade' 做多（接近低點、會賺）；(2) 中午以後突破 28,5xx 的爆量長紅 → 'follow' 做多，搭上往 28,697 的一段。",
    needed_data=["NQ 1 分 K 含成交量", "前 20 日同時段 5 分 K 量與幅度"],
))

S.append(dict(
    id="Q15", name="趨勢日內 RSI(2) 回檔順勢進場", family="盤中趨勢＋短期超賣",
    priority="中高（把『趨勢日延續』拆成多個小停損的回檔進場點，目標是 A2/Q03 的低回撤版）",
    entry_window_et="10:30～14:30，每根 5 分 K 收盤判斷；15:29 前出場",
    definitions={
        "D": "趨勢方向：|C_t − O930| ≥ a×ADR10 時 D = sign(C_t − O930)，否則 D=0（不做）",
        "VWAP 側": "D=+1 時要求 C_t > VWAP_t；D=−1 時要求 C_t < VWAP_t（VWAP 只用來判斷方向，不是帶）",
        "RSI2": "5 分 K 收盤的 2 期 Wilder RSI（從 09:30 第一根起算，跨日不延續）",
    },
    entry="10:30 起到 14:30，若 D=+1、價格在 VWAP 上方、RSI2 ≤ L → 下一分鐘做多；D=−1、在 VWAP 下方、RSI2 ≥ 100−L → 做空。每天最多 2 筆，不同時持有兩筆。",
    stop="SL = min(145, 0.25×ADR10)。",
    exit="TP = tp_R × SL；否則 15:29（exit_slot 1289）收盤出場。",
    params_to_explore={"L": [5, 10, 20], "a": [0.35, 0.5, 0.75], "tp_R": [1.0, 2.0]},
    params_fixed={"start": "10:30", "end": "14:30", "max_trades_per_day": 2, "exit_slot": 1289},
    rationale="趨勢日（已走出大半個平均日振幅）的方向傾向延續（A2、Q03），但在延伸點追價停損很大；等短期超賣（RSI2 極端）的回檔再進場，用較小的停損參與同一個趨勢——Connors 的『順大勢、逆小勢』。",
    evidence_sources=[
        {"cite": "Connors & Alvarez (2009), Short Term Trading Strategies That Work（RSI(2) 在上升趨勢中買超賣）",
         "url": "https://www.mql5.com/en/articles/17636",
         "finding": "日線：大趨勢向上時 RSI(2) 極低買進有長期正報酬；MQL5 文章把同規則放到 30 分 K 測試。"},
        {"cite": "本專案 A2_TREND_EXT_FOLLOW（內部）",
         "url": "（research/agents/A2/REPORT.md）",
         "finding": "延伸 ≥0.75×ADR10 後順勢，三段每筆皆正；但回撤 −$16k 以上。"},
        {"cite": "Zarattini, Aziz & Barbon (2024) SFI RP 24-97",
         "url": "https://concretumgroup.com/beat-the-market-an-effective-intraday-momentum-strategy-for-sp500-etf-spy/",
         "finding": "走出時段雜訊帶後的趨勢延續（Q03 的同一機制）。"},
    ],
    critiques=[
        "RSI(2) 的實證主要在日線；盤中 5 分 K 的證據很少（MQL5 30 分 K 測試結果不明）。",
        "回檔有時就是反轉的開始（趨勢在午盤結束），停損 0.25×ADR10 可能常被打；需和『同條件但不等回檔、直接追』比較。",
        "趨勢判斷用 O930，跳空大的日子 D 可能與日線方向相反。",
    ],
    expected_1111="O930 = 28,352.75；ADR10 約 450～550 → a=0.5 的門檻約 ±225～275 點 → 價格 ≥ 28,578～28,628 時 D=+1（當天高點 28,697，有達到；早盤低點 28,199 只離開盤 −154，不足以形成 D=−1）。之後只要 5 分 RSI2 ≤ 10 且仍在 VWAP 上方就做多，例如在 28,62x～28,65x 附近；停損約 110～140 點。若高點後一路回落到 28,484（收盤）→ 停損機率高（−$2,200～−$2,800）；若回檔後再創 28,697 → 1R 止盈。結果取決於高點出現的時間，偏不利。a=0.75（約 ±340～410）時不會觸發。",
    needed_data=["NQ 1 分 K（聚合成 5 分 K）", "成交量（VWAP）", "ADR10"],
))

# ---------------------------------------------------------------------------------------------
for s in S:
    n = 1
    for v in s["params_to_explore"].values():
        n *= len(v)
    s["n_variants"] = n
    assert len(s["params_to_explore"]) <= 3
    assert all(2 <= len(v) <= 4 for v in s["params_to_explore"].values())
    assert n <= 100

TOTAL = sum(s["n_variants"] for s in S)

json_out = {
    "author": "P2_quant",
    "date": "2026-11-12",
    "note": "研究用途，不是投資建議。只寫規格，未回測。所有時間為美東（ET）；slot = (分鐘數 − 1080) mod 1440。",
    "common_rules": COMMON.strip(),
    "event_calendar": {"FOMC": FOMC, "CPI": CPI, "NFP": NFP, "note": CAL_NOTE},
    "total_variants": TOTAL,
    "strategies": [{k: s[k] for k in ["id", "name", "family", "priority", "entry_window_et", "definitions", "entry", "stop",
                                       "exit", "params_to_explore", "n_variants", "params_fixed", "rationale",
                                       "evidence_sources", "critiques", "expected_1111", "needed_data"]} for s in S],
}
with open(os.path.join(OUT, "strategies.json"), "w", encoding="utf-8") as f:
    json.dump(json_out, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------------------------
L = []
L.append("P2_quant：NQ 日內「量化／統計」策略規格（15 個）")
L.append("研究用途，不是投資建議。只寫規格，沒有回測。日期 2026-11-12。")
L.append("不重複另一位撰寫者的價格結構類（ICT/SMC/SNR/破底翻/Turtle Soup/ORB/VWAP 帶/Market Profile/IB/Asian range）。")
L.append(COMMON)
L.append("==========================================================================================")
L.append("總覽（依建議的回測優先順序）")
L.append("==========================================================================================")
order = ["Q03", "Q13", "Q15", "Q02", "Q05", "Q14", "Q11", "Q01", "Q10", "Q06", "Q07", "Q08", "Q09", "Q04", "Q12"]
byid = {s["id"]: s for s in S}
for k, sid in enumerate(order, 1):
    s = byid[sid]
    L.append(f"{k:2d}. {sid} {s['name']}｜{s['family']}｜變體 {s['n_variants']}｜優先：{s['priority']}")
L.append(f"\n合計 {TOTAL} 種變體（15 個策略）。")
L.append("""
與本專案已做過的研究的關係（請挑戰者特別注意）：
- A2 已在 NQ 2024–25 看過時段/日曆效應：Gao 首→末半小時動能 ≈0、15:30 尾盤反轉 validation 翻負、
  10:00 雙向反轉不穩、過度延伸反向 18 組全虧、月初月底/星期/OPEX 描述性 t<3。
  → Q01、Q02 是『條件版』，先驗偏低；Q05 只做多並以時段 σ 標準化；Q10 加入早盤回落條件。
- A2_TREND_EXT_FOLLOW 與 B3_NQES_RS_0945_T120 是唯二『三段都正』的乾淨線索，但回撤大。
  → Q03（時段雜訊帶＋VWAP 追蹤）、Q15（趨勢日回檔小停損）、Q13（z 標準化＋停損）都是針對『降低回撤』的改寫。
- B2：CPI/NFP 08:30 開盤區間突破、非事件日跳空回補 validation 失敗。→ Q09、Q04 先驗低，列在最後。

事件過濾（不是獨立策略，建議回測者在所有策略的結果上分組報告）：
- FOMC 日（14:00 後波動大）、CPI/NFP 日、每月選擇權到期日（第三個週五；季月為四巫日）、
  換月週（到期前 8 個日曆日起）、美國公債休市但股市開盤的日子（例如 11-11 退伍軍人節、10 月哥倫布日，債券期貨流動性低）。
- 文獻：Elms (2026, SSRN) 用 2016–2025 標普期權發現『沒有到期日 pinning，近到期 ATM 未平倉量高時日區間反而更寬』；
  SpotGamma 等實務認為到期後一週波動放大（未經驗證）。所以 OPEX 只當分組變數，不做方向假設。

多重檢定：
- 15 個策略 208 種變體。建議：每個策略先在『探索』選 1 組（要求相鄰參數也賺＝高原），
  驗證期只跑這 15 組，用 Holm 校正（15 個假設）；測試期只報告。
- 事件型（Q06–Q09）樣本太小，需用排列檢定（隨機挑同樣數量的非事件日、同時段、同規則）當基準。
- 只做多的策略（Q05、Q06、Q08、Q10、Q11）必須和『同時段無條件做多』比較，否則只是吃到多頭漂移。
""")
L.append("==========================================================================================")
L.append("詳細規格")
L.append("==========================================================================================")

def fmt_val(v):
    if isinstance(v, list):
        if len(v) > 12:
            return "[" + ", ".join(map(str, v[:6])) + f", …（共 {len(v)} 個，完整清單見 JSON）]"
        return "[" + ", ".join(map(str, v)) + "]"
    return str(v)

for sid in order:
    s = byid[sid]
    L.append("")
    L.append("------------------------------------------------------------------------------------------")
    L.append(f"{s['id']}  {s['name']}")
    L.append("------------------------------------------------------------------------------------------")
    L.append(f"家族：{s['family']}")
    L.append(f"優先：{s['priority']}")
    L.append(f"時間窗（美東）：{s['entry_window_et']}")
    L.append("定義：")
    for k, v in s["definitions"].items():
        if k in ("數據日", "日曆說明", "FOMC 日"):
            v = (v[:160] + "…（完整清單見 JSON 的 event_calendar）") if len(v) > 160 else v
        L.append(f"  - {k}：{v}")
    L.append(f"進場：{s['entry']}")
    L.append(f"停損：{s['stop']}")
    L.append(f"出場：{s['exit']}")
    L.append(f"要探索的參數（{s['n_variants']} 種）：")
    for k, v in s["params_to_explore"].items():
        L.append(f"  - {k} ∈ {fmt_val(v)}")
    L.append("固定參數：")
    for k, v in s["params_fixed"].items():
        if k in ("calendar_note",):
            continue
        L.append(f"  - {k} = {fmt_val(v)}")
    L.append(f"交易邏輯：{s['rationale']}")
    L.append("文獻與證據：")
    for e in s["evidence_sources"]:
        L.append(f"  * {e['cite']}")
        L.append(f"    {e['finding']}")
        L.append(f"    來源：{e['url']}")
    L.append("已知弱點／失效情境：")
    for c in s["critiques"]:
        L.append(f"  - {c}")
    L.append(f"2026-11-11 會怎麼做：{s['expected_1111']}")
    L.append(f"需要的資料：{'、'.join(s['needed_data'])}")

L.append("")
L.append("==========================================================================================")
L.append("事件日曆（2024–2026）")
L.append("==========================================================================================")
L.append("FOMC（聲明日，14:00 ET）：" + ", ".join(FOMC))
L.append("CPI（08:30 ET）：" + ", ".join(CPI))
L.append("NFP（08:30 ET）：" + ", ".join(NFP))
L.append(CAL_NOTE)
L.append("""
==========================================================================================
研究限制
==========================================================================================
- 這個環境的 WebFetch 連不到 nd.edu、unisg.ch、mdpi.com、bls.gov、newyorkfed.org 等網域，
  論文細節（例如 Noise Area 的精確公式、SSRN 4131740 的作者）來自搜尋摘要與二手整理，已標註；
  建議回測者若能連網，再核對原文。
- 2026-11-11 的推演只用使用者給的幾個價格（開 28,352.75、低 28,199、高 28,697、收 28,484；
  前日高 28,090、低 27,516、收 27,835.75；夜盤高約 28,398、低約 27,848），時間細節不明的地方都寫成條件句。
""")
with open(os.path.join(OUT, "strategies.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("ok", TOTAL, len(S))
