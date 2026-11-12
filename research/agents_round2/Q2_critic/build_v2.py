"""Q2_critic：由 P2_quant/strategies.json 產生修正版 strategies_v2.json。研究用途，不是投資建議。"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "P2_quant", "strategies.json")
src = json.load(open(SRC, encoding="utf-8"))
old = {s["id"]: s for s in src["strategies"]}

COMMON = """修正版共用規則（Q2_critic；未列出的沿用 P2 的共用定義）
- slot(hh:mm) = (hh*60+mm − 1080) mod 1440。『hh:mm 的價格』＝ slot(hh:mm)−1 那根 1 分 K 收盤；NaN 時往前找最後一根有值的收盤（最多 3 分鐘，否則當天這個訊號不做）。
- 進場價＝entry_slot 的開盤；NaN 時用之後 3 分鐘內第一根有值的開盤，否則不做。
- 『前 N 天同時刻』『前 N 天』：一律只用 [i−N, i−1]（不含當天）；有效值 < 0.75×N 當天不做。
- 日報酬（σd20 等）只用『前一日與今日同合約』的配對（g.prev_same_contract），取最近 20 個有效值；不可用連續合約直接相除（會吃進換月價差）。
- ADRn＝前 n 個 RTH 日（09:30–15:59）高低差平均，不含當天。
- VWAP_T＝slot 930 到 slot(T)−1（含）的 Σ((H+L+C)/3×V)/ΣV，量為 NaN 或 0 的分鐘略過。
- 停損：1 分 K 高低點觸價即出（lib.sim 做法）；跳空穿越用開盤價；同一分鐘止盈止損都碰到算止損。『條件出場』（追蹤、時間）只在規定的檢查點用收盤判斷、下一根開盤出場。時間出場＝exit_slot 那根收盤。
- 停損上限 145 點（1 口最壞 −$2,948）。每日 $3,000：今日已實現（平倉即記帳，Pine 式）− 這筆最壞 − 其他持倉最壞 < −3,000 就不下。
- 同一分鐘多個訊號的固定順序（現在事先登記）：基準（順勢多點）＞ C1 ＞ TD10 ＞ Q03v2 ＞ Q13v2 ＞ Q02v2 ＞ Q08v2 ＞ EVT_FOMC。
- 換月日（prev_same_contract 為 False）：用到 PDC／PDH／PDL／跳空／日報酬的策略當天不做。
- 部位：每個策略報 (a) 單獨 1 口；(b) 0.5 口（5 MNQ）併入基準、先到先做。1 口 145 點停損的單在基準持倉時必定被每日檢查擋掉。
- 必報壓力（不可用來挑參數）：$48；進出各差 2 點＋$48；成本 $96；10:00、15:30、事件時刻進場者另報『晚 2 分鐘進場』。
- 必報分組（不可據以事後加過濾）：FOMC 日、CPI/NFP 日、OPEX（第三個週五）、換月週、債市休市股市開、TOM（月末最後 1 日＋月初 3 日）、IBS 三桶、NR7/ID、RVR（RV5/RV60）三桶；和 C1、TD10 同日／不同日。
- 挑選：每個策略只在探索期挑 1 組（相鄰參數也要賺＝高原），驗證期只跑這 5 組，Holm 校正 5 個假設；測試期、近期只報告。
- 測試期汙染標記：Q03v2、Q13v2（09:45）屬於已在 2026 被看過的『開盤／盤中動能』家族（A2_TREND +206、B3 +314、A_trend F1 家族 75% 測試賺），測試期通過不是獨立證據；真正乾淨的只有 2026-11-12 起的前進測試。
- 資料：Yahoo 期間（10/20 後）RTH 量的口徑比之前高 34%～40%，夜盤缺 5.7% 分鐘；事件日曆已用 08:30／14:00 那根 1 分 K 放大倍數核對（CPI 34、NFP 34 全部 ≥3 倍；FOMC 23 全部 ≥2.1 倍）。
"""

def ev(sid, idxs=None):
    e = old[sid]["evidence_sources"]
    return e if idxs is None else [e[k] for k in idxs]

strategies = []

# ---------------- Q03v2 ----------------
strategies.append({
    "id": "Q03v2", "from": ["Q03"], "priority": 1,
    "name": "Noise Area 時段雜訊帶突破（Zarattini–Aziz–Barbon）修正版",
    "family": "盤中趨勢／波動狀態",
    "status": "回測（測試期非獨立，見 common_rules）",
    "entry_window_et": "第一個檢查點 10:00 或 10:30，之後每半點檢查到 15:30；15:59 收盤前出場",
    "definitions": {
        "move_d(T)": "|C[d, slot(T)−1] / O[d,930] − 1|",
        "σ_T": "前 14 個 RTH 日 move_d(T) 的平均（同一時刻、不含當天）",
        "UB_T": "max(O930, PDC) × (1 + m×σ_T)",
        "LB_T": "min(O930, PDC) × (1 − m×σ_T)",
        "VWAP_T": "見 common_rules（09:30 到 slot(T)−1）",
        "檢查點": "start, start+30 分, …, 15:30（start=10:00 時 12 個點，10:30 時 11 個點）",
    },
    "entry": "在檢查點 T：C[slot(T)−1] > UB_T → slot(T) 開盤做多；< LB_T → 做空。每天最多 2 筆、同時只持 1 筆。出場後只能從『出場之後的下一個檢查點』起重新判斷（被追蹤出場的那個檢查點不能立刻再進）；第二筆方向可不同。",
    "stop": "硬停損＝min(145, 0.25×ADR10) 點，1 分 K 高低點觸價即出（任何時間）。",
    "exit": "trail=vwap_band：進場之後的每個檢查點 T'（不含進場那個），多單若 C[slot(T')−1] < max(UB_T', VWAP_T') → slot(T') 開盤出場；空單若 > min(LB_T', VWAP_T') 出場。trail=hard_only：只有硬停損。都沒觸發 → 15:59 收盤（exit_slot 1319）。不設止盈。",
    "params_to_explore": {"m": [1.0, 1.5, 2.0], "trail": ["vwap_band", "hard_only"], "start": ["10:00", "10:30"]},
    "n_variants": 12,
    "params_fixed": {"N_lookback_days": 14, "hard_stop": "min(145, 0.25*ADR10)", "max_trades_per_day": 2,
                     "exit_slot": 1319, "check_interval_min": 30, "skip_roll_day": True},
    "controls_report_only": [
        "start=10:00 版本另報『10:02 開盤進場』壓力（RT：TD10 探索期 44% 優勢來自 10:00→10:02）",
        "依 C1／TD10 當天有沒有單拆開報告每筆；報告同日重疊數與方向一致率",
        "RVR 三桶分組（原 Q03 的可選過濾，只報告）",
    ],
    "changes": [
        "硬停損由 min(145, |進場−對側邊界|) 改為 min(145, 0.25×ADR10)：原定義的對側邊界距離 2mσ＋跳空，幾乎永遠 >145，等於每筆都 145 點、最壞 −$2,948 且和基準互斥。0.25×ADR10 是 A_trend 掃 29,475 種時普遍較好的止損，也是 C1 用的。",
        "刪掉 N_lookback {10,14,20}，固定 14（論文值）；N 只是換雜訊估計，低價值參數。",
        "新增 start {10:00, 10:30}：10:00 檢查碰到數據公布分鐘、且和 TD10／C1 同日同向高度重疊；10:30 起算才是乾淨的增量。",
        "寫死再進場規則（出場後下一個檢查點起）、追蹤只在進場之後的檢查點、硬停損任何時間觸價。",
        "VWAP 範圍、σ_T 不含當天、缺分鐘處理寫明。",
    ],
    "rationale": old["Q03"]["rationale"],
    "evidence_sources": ev("Q03"),
    "critiques": [
        "和 A_trend F4（≈0）、F6b 趨勢日（測試 −350）、C4 VWAP 單邊（測試 −378）是近親；A_trend 也發現『跌破 VWAP 就出』多半太早出——vwap_band 的主要風險，所以 hard_only 對照必測。",
        "整個盤中延伸順勢家族的 2026 結果已知（A2_TREND +206），測試期通過不算獨立證據。",
        "勝率約 35%～50%、靠少數趨勢日；1 口版和基準搶每日額度。",
        "論文的績效有很大部分來自波動目標槓桿，固定 1 口的 NQ 會差很多。",
    ],
    "expected_1111": "實際資料示意：σ_T 約 0.74%～1.11%。m=1：11:30 收 28,568.5 > UB 28,561.4 → 11:30 開盤 28,568.75 做多；13:00 收 28,632.5 < UB 28,633.1 → 13:00 開盤 28,633 出場 +64 點（+$1,237）；13:30 再越界做多 28,680.75，最低 28,535.0 碰到 145 點停損 −$2,948；當天約 −$1,711。hard_only：11:30 做多抱到 15:59 收 28,484.25，約 −$1,738。m=1.5、2.0 不交易。start 10:00／10:30 結果相同（10:00、10:30 都在帶內）。",
    "needed_data": ["NQ 1 分 K（含量）", "前 14 日同時刻離開盤報酬", "ADR10", "PDC（同合約）"],
})

# ---------------- Q13v2 ----------------
strategies.append({
    "id": "Q13v2", "from": ["Q13"], "priority": 2,
    "name": "NQ 相對 ES 強弱（z 分數）順勢，修正版",
    "family": "跨市場／相對強弱",
    "status": "回測（09:45 版測試期非獨立；13:00 版是新的）",
    "entry_window_et": "09:45 或 13:00 決策，持有 120 分鐘",
    "definitions": {
        "β60": "前 60 個 RTH 日（不含當天）NQ 開→收報酬對 ES 開→收報酬的 OLS 斜率",
        "rel_T": "ln(NQ C[s−1]/C[s−61]) − β60 × ln(ES C[s−1]/C[s−61])，s=slot(T)（09:45：08:44→09:44；13:00：11:59→12:59）",
        "z_T": "rel_T / std(前 20 日同一決策時刻的 rel)（不含當天，至少 15 個有效值）",
    },
    "entry": "z_T ≥ z0 做多 NQ；z_T ≤ −z0 做空（z0=0 時看正負號，z=0 不做）。slot(T) 開盤進場，每天最多 1 筆。NQ 或 ES 任一邊在 s−1、s−61 缺值（含往前 3 分鐘補值仍缺）就不做。",
    "stop": "SL＝min(145, 0.35×ADR10)，1 分 K 高低點觸價。",
    "exit": "entry_slot+119 那根收盤（09:45 → 11:44 收；13:00 → 14:59 收）。不設止盈。",
    "params_to_explore": {"decision_time": ["09:45", "13:00"], "z0": [0.0, 1.0, 1.5]},
    "n_variants": 6,
    "params_fixed": {"mode": "mom", "lookback_min": 60, "hold_min": 120, "beta_window_days": 60, "z_window_days": 20},
    "controls_report_only": [
        "B3 原版（09:45、正負號、無停損、11:44 出）",
        "依 C1／TD10 當天有沒有單拆開報告；09:45 版和基準 I2 同日重疊數",
    ],
    "changes": [
        "刪掉 mode='rev'：它是 mom 的鏡像，B3 已在探索＋驗證確認 mom 方向，再測 rev 只會把多重檢定次數翻倍。",
        "刪掉 10:30：B3 表格（探索＋驗證）10:30 NQ−ES 120 分 驗證 −65。",
        "z0 改成 {0, 1.0, 1.5}，0 對應 B3 原版加停損，方便和已知結果銜接。",
        "寫明 rel 的分鐘、z 與 β 不含當天、缺值不做；Yahoo 期間兩市場同源，缺任一邊就不做。",
    ],
    "rationale": old["Q13"]["rationale"],
    "evidence_sources": ev("Q13", [0, 1, 3]),
    "critiques": [
        "B3（2026 +314）、A_trend C2（測試 +126、近期 −668、回撤 −36k）已看過 2026：09:45 版測試期不是獨立證據。",
        "停損 0.35×ADR10 在 2026 幾乎都頂到 145，1 口和基準互斥；看 0.5 口版。",
        "大型科技財報日 NQ/ES 價差可能持續偏離（對 mom 有利）也可能急反轉；按財報週分組報告。",
    ],
    "expected_1111": "實際資料示意：β60＝1.35。09:45：rel −9.7bp、z −0.36 → z0=0 於 09:45 開盤（約 28,288）做空，之後漲過 28,433 停損 −$2,948；z0≥1 不做。13:00：rel −12.0bp、z −1.42 → z0=0、1.0 於 13:00 開盤 28,633 做空，最高 28,697 未碰停損，14:59 收 28,552.25 出場 +80.75 點（+$1,567）；z0=1.5 不做。",
    "needed_data": ["NQ、ES 1 分 K（同一分鐘對齊）", "前 60 日 RTH 開收", "前 20 日同時刻 rel", "ADR10"],
})

# ---------------- Q02v2 ----------------
strategies.append({
    "id": "Q02v2", "from": ["Q02", "Q01"], "priority": 3,
    "name": "大位移日（負 gamma 代理）尾盤 15:30 順勢，修正版（Q01 併入）",
    "family": "時段季節性／盤中動能",
    "status": "回測",
    "entry_window_et": "15:30 進場，15:59 收盤出場",
    "definitions": {
        "ROD": "C[i,1289] / PDC − 1（PDC 必須同合約；換月日不做）",
        "σd20": "前 20 個有效『同合約』日報酬的標準差（不含當天）",
    },
    "entry": "若 |ROD| ≥ k×σd20，於 slot 1290（15:30）開盤順著 sign(ROD) 進場；每天最多 1 筆。",
    "stop": "固定 SL ∈ {60, 120} 點，1 分 K 高低點觸價。",
    "exit": "15:59 收盤（exit_slot 1319）。不設止盈。",
    "params_to_explore": {"k": [0.5, 1.0, 1.5], "SL_pts": [60, 120]},
    "n_variants": 6,
    "params_fixed": {"entry_slot": 1290, "exit_slot": 1319, "contracts": 1},
    "controls_report_only": [
        "k=0（每天 15:30 順著 ROD）＝涵蓋原 Q01 的無條件版",
        "反向版（A2_CLOSE_REVERSAL_1530 的近似）",
        "依 C1／TD10 當天有沒有單拆開（大位移日≈趨勢日，有加倍下注的問題）",
        "OPEX、月底（再平衡）分組",
    ],
    "changes": [
        "刪掉 entry_time 15:45：只剩 14 分鐘、跨 15:50 收盤失衡公布，邊際比成本小。",
        "SL {50,100} → {60,120}：探索期 15:30 多單最大不利 ≥40 點 32%、≥60 點 18%，50 點約四分之一被雜訊掃。",
        "σd20 只用同合約日報酬，避免換月價差灌大 σ（連續合約相除會吃進 150～300 點價差）。",
        "Q01 刪除併入：r1（前收→10:00）和 ROD 高度同號、同一時段同一方向，算同一個假設；A2 已測 Gao 效應 ≈0。",
    ],
    "rationale": old["Q02"]["rationale"],
    "evidence_sources": ev("Q02") + [old["Q01"]["evidence_sources"][0], old["Q01"]["evidence_sources"][3]],
    "critiques": [
        "k=0 是 A2 尾盤反轉（探索 +121／驗證 −15）的反方向，無條件版預期探索期為負；價值只在大位移條件。",
        "沒有 GEX 資料，|ROD| 只是負 gamma 代理。",
        "k=1.5 每段筆數少（探索約 55～70 筆）。",
    ],
    "expected_1111": "實際資料示意：ROD +2.45%、σd20 2.19% → 1.12σ。k=0.5、1.0 於 15:30 開盤 28,517.25 做多；15:30～15:59 最低 28,443.5：SL60 被停損 −$1,248；SL120 抱到 15:59 收 28,484.25 → −33 點（−$708）。k=1.5 不做。",
    "needed_data": ["NQ 1 分 K", "PDC（同合約）", "前 20 個同合約日報酬"],
})

# ---------------- Q08v2 ----------------
strategies.append({
    "id": "Q08v2", "from": ["Q08"], "priority": 4,
    "name": "CPI∪NFP 公布前夜盤做多（合併事件），修正版",
    "family": "事件日／夜盤",
    "status": "回測＋排列檢定；樣本小，通過門檻較嚴",
    "entry_window_et": "數據日夜盤 03:00 或 06:00 進場，08:28 收盤出場（08:30 公布前）",
    "definitions": {
        "數據日": "event_calendar 的 CPI ∪ NFP（合併；探索 36、驗證 10、測試 18、近期 4）。股市休市日（如 2026-04-03）若網格有夜盤仍可交易，否則跳過。",
    },
    "entry": "數據日在 entry_time（slot 540 或 720）開盤做多（缺分鐘照 common_rules）。每天 1 筆。",
    "stop": "固定 SL ∈ {80, 120} 點。",
    "exit": "08:28 收盤（exit_slot 868）。不設止盈。",
    "params_to_explore": {"entry_time": ["03:00", "06:00"], "SL_pts": [80, 120]},
    "n_variants": 4,
    "params_fixed": {"event_set": "CPI+NFP", "exit_slot": 868, "direction": "long"},
    "controls_report_only": [
        "同規則套在所有非數據日（一般隔夜漂移）——必須贏它才算事件效果",
        "排列檢定：同一分割內隨機抽同樣天數的非數據日，10,000 次，報 p 值",
        "CPI、NFP 各自的結果只報告，不准據以挑選",
        "夜盤壓力：進出各差 2 點",
    ],
    "pass_rule_extra": "除了 BRIEF 的三段都賺，還要：探索期排列 p < 0.10，且探索、驗證都贏非數據日對照。",
    "changes": [
        "event_set 不再拆 CPI／NFP／合併三選一，固定合併：把 70 個事件拆半再挑是資料挖掘。",
        "SL {40,80} → {80,120}：探索期 03:00 進場多單最大不利 ≥40 點 54%、≥60 點 41%、≥100 點 19%；06:00 ≥40 點 33%。40 點等於擲銅板。",
        "新增非數據日對照與排列檢定為必報、通過門檻加嚴。",
        "修正 P2 的成本說法：10bp ≈ 25 點 ≈ $500，相對 $48 不算薄；問題是文獻效果不確定。",
        "日曆已核對：68 個 CPI/NFP 日 08:30 那根都 ≥3 倍放大。",
    ],
    "rationale": old["Q08"]["rationale"],
    "evidence_sources": ev("Q08"),
    "critiques": [
        "證據是工作論文，Lucca–Moench 反面；Liberty Street 2026 指出隔夜漂移在 2021–25 消失。",
        "夜盤流動性較差，Yahoo 期間夜盤缺 5.7% 分鐘。",
        "樣本 36／10／18 筆，信賴區間很寬；就算通過也建議先紙上記錄。",
    ],
    "expected_1111": "11-11 不是數據日 → 不交易（CPI 11-10、NFP 11-06）。",
    "needed_data": ["NQ 1 分 K（夜盤）", "CPI／NFP 日曆（已核對）"],
})

# ---------------- EVT_FOMC ----------------
strategies.append({
    "id": "EVT_FOMC", "from": ["Q06", "Q07"], "priority": 5,
    "name": "FOMC 事件家族：公布前漂移＋記者會後反轉（固定規則，只紙上記錄）",
    "family": "事件日",
    "status": "只紙上記錄／描述統計；不挑參數、不算『通過』",
    "entry_window_et": "FOMC 當天：規則 A 09:31→13:55；規則 B 15:00→15:59",
    "definitions": {
        "FOMC 日": "event_calendar.FOMC（探索 12、驗證 4、測試 6、近期 1）",
        "Δ": "C[i,1259] − C[i,1199]（13:59 收 → 14:59 收）",
    },
    "entry": "規則 A（原 Q06）：FOMC 當天 slot 931 開盤做多。規則 B（原 Q07）：|Δ| ≥ 0.15×ADR10 時，slot 1260（15:00）開盤逆著 Δ 進場。",
    "stop": "規則 A：SL 120；規則 B：SL 100。",
    "exit": "規則 A：13:55 收盤（exit_slot 1195），不碰 14:00 公布；規則 B：15:59 收盤（exit_slot 1319）。",
    "params_to_explore": {"rule": ["A_preFOMC", "B_postPC_reversal"]},
    "n_variants": 2,
    "params_fixed": {"A_entry_slot": 931, "A_exit_slot": 1195, "A_SL": 120, "include_eve": False,
                     "B_q": 0.15, "B_entry_slot": 1260, "B_exit_slot": 1319, "B_SL": 100},
    "controls_report_only": [
        "排列檢定：同一分割內隨機抽同樣數量的非 FOMC 日、同時段同規則（規則 B 用同樣的 Δ 定義與門檻），10,000 次",
        "規則 A 另和『所有非 FOMC 日 09:31→13:55 做多』比較",
        "規則 B 另報『晚 2 分鐘進場』與進出各差 4 點（記者會尾段波動大）",
    ],
    "changes": [
        "Q06、Q07 合併成一個事件家族，共 2 條事前固定規則、0 個要挑的參數（原 12＋8＝20 種）。FOMC 驗證期只有 4 次，不可能挑參數。",
        "Q06 固定 09:31 進、SL 120、不做前一天。",
        "Q07 的進場由 14:45（記者會進行中，1 分 K 常 30～80 點）改成 15:00，Δ 改量到 14:59，不浪費 14:45～14:59 的資訊；q 固定 0.15、SL 100。",
        "狀態改為只紙上記錄：23 次事件不可能得出統計結論。",
    ],
    "rationale": old["Q06"]["rationale"] + "／" + old["Q07"]["rationale"],
    "evidence_sources": ev("Q06") + ev("Q07"),
    "critiques": [
        "樣本 12／4／6／1；即使效果存在，標準誤也遠大於效果。",
        "pre-FOMC drift 2015 年後衰減；記者會後反轉的證據以 2011–2022 為主。",
    ],
    "expected_1111": "不是 FOMC 日（上次 10-28、下次 12-09）→ 不交易。",
    "needed_data": ["NQ 1 分 K", "FOMC 日曆（已核對）", "ADR10"],
})

deleted = {
    "Q01": "A2 已測 Gao 首→末半小時 ≈0；和 Q02 同一假設（併入 Q02v2 的 k=0 對照）；SL 40 約三分之一被雜訊掃。",
    "Q04": "B2_GAPF 驗證 −19、C_indicators 缺口回補測試 −710；是 C6 跳空延續（三段都賺）的反方向；和 I2 重疊。",
    "Q05": "和 TD10／C1（開盤動能延續）正面衝突；A2 10:00 反轉、過度延伸反向、C 的價值區外回歸都失敗；10:00 數據分鐘滑價。",
    "Q09": "B2 08:30 ORB 驗證 −1,276、A_trend F5a、C『夜盤報酬順勢』、BRIEF『夜盤大漲後追多』失敗家族的子集；cont/fade 找符號。改成分組變數。",
    "Q10": "A2 月初月底 t<3；RTH 開→收做多 ≈+21／筆，TOM 效果多在隔夜；10:00_dip 同 Q05 問題。改成分組變數。",
    "Q11": "C『前一天漲跌反向』『日線 RSI 反向』測試期失敗，同機制；IBS 效果多在隔夜。改成分組變數。",
    "Q12": "突破家族（A_trend F2、F4）失敗；證據只有實務書；樣本小、NR7/ID 重疊。改成分組變數。",
    "Q14": "C／B3／FINAL 都顯示量能不預測方向；Yahoo 期間量口徑高 34%～40%，RVOL 失真；follow/fade 找符號；每天 3 筆搶額度。",
    "Q15": "A_trend F3 回檔進場幾乎全賠（兩段都賺者測試只 5% 賺）、F6b、C4 失敗；指標過濾測試期 0 改善；tp_R=1 屬縮小止盈失敗類。",
}

out = {
    "author": "Q2_critic",
    "date": "2026-11-12",
    "based_on": "P2_quant/strategies.json（15 個、208 種）",
    "note": "研究用途，不是投資建議。修正後要回測的最終規格；priority 1＝最值得測。只審規格，沒有回測任何損益。",
    "common_rules": COMMON,
    "event_calendar": src["event_calendar"],
    "total_variants": sum(s["n_variants"] for s in strategies),
    "multiple_testing": "5 個策略、30 種組合；每策略探索期挑 1 組（高原），驗證期只跑 5 組，Holm 校正 5 個假設；測試／近期只報告。EVT_FOMC 不參與『通過』判斷。對照組只報告、不參與挑選。",
    "kept_from_original": ["Q02", "Q03", "Q06", "Q07", "Q08", "Q13"],
    "deleted": deleted,
    "grouping_variables_report_only": ["FOMC 日", "CPI/NFP 日", "OPEX", "換月週", "債市休市股市開", "TOM", "IBS 三桶", "NR7/ID", "RVR 三桶", "C1/TD10 同日"],
    "strategies": strategies,
}
assert out["total_variants"] <= 150
for s in strategies:
    n = 1
    for v in s["params_to_explore"].values():
        n *= len(v)
    assert n == s["n_variants"], s["id"]
    assert len(s["params_to_explore"]) <= 3
json.dump(out, open(os.path.join(HERE, "strategies_v2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok", out["total_variants"], [(s["id"], s["priority"], s["n_variants"]) for s in strategies])
