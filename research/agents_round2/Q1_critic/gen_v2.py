"""產生 strategies_v2.json（Q1_critic 修正版規格）。研究用途，不是投資建議。"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "..", "P_strategist", "strategies.json")))
SRC = {s["id"]: s["sources"] for s in P["strategies"]}

COMMON = [
    "沿用 P_strategist 的 C1～C15（時間、K 棒對齊、ATR／ADR、轉折點、FVG、OB、流動性池、掃蕩、BOS／CHoCH、停損上限、每日風控、筆數），以下 QC1～QC10 優先。",
    "QC1 真限價（取代 C9）：多單掛 P，在掛單有效期間（且不在 09:58～10:02）第一個 slot s 滿足 L[s] <= P−0.25 才算成交；成交價＝min(P, O[s])。成交那分鐘 L[s] <= 停損價 → 算停損出場；若 O[s] 已經 <= 停損價 → 記完整虧損 (P−停損)。成交那分鐘不算止盈。空單相反。不再有『O[s+1] 在停損外就取消』（那會刪掉最差的單）。P 的 C9 近似只當對照另報。",
    "QC2 停損：結構停損 sl_pts < 15 → 用 15；> 145 → 這筆不做（不截斷）。報告兩種情況的筆數。每筆最壞 145×20+48＝$2,948。",
    "QC3 10:00 數據：09:58～10:02（slot 958～962）不進場。市價訊號的進場 slot 落在這段 → 改在 10:03（slot 963）開盤進場，前提是到 10:02 收盤為止價格沒碰到停損價或止盈價；否則取消。限價單這 5 分鐘不算觸價。",
    "QC4 出場 B55：止盈 +55、止損 −145（固定點數，不用結構停損）。時間出場：進場 < 12:00 → 12:29（slot 1109）；進場 >= 12:00 → 15:49（slot 1309）。訊號集合和結構版相同（結構停損 >145 的訊號兩版都不做），比的是同一批訊號。",
    "QC5 VWAP 同側：VWAP＝09:30 起 1 分 K 典型價 (H+L+C)/3 的量加權，算到判斷那一分鐘（含）。做多要求判斷那一分鐘收盤 > VWAP，做空 < VWAP。市價訊號用訊號 K 最後一分鐘；限價單用觸價前一分鐘（s−1）的收盤。",
    "QC6 ES 同步：在 ES 網格（data.load_grid('ES')）用同一條規則找同方向事件，所有點數參數 ×0.28（四捨五入到 0.25，最小 0.25 點）；ES 事件確定時間 <= NQ 事件確定時間（同一根 K 或更早、同一天）。ES 缺值或沒有事件 → 不同步。",
    "QC7 換月：多日價位（N 日高低、2 日回看水平）只用和當天同一合約的日子（g.contract）；回看期跨換月 → 當天這條規則不做，報告被刪的天數。ADR10 是振幅，不受影響。",
    "QC8 重疊報告：每筆標記 base_same_day_dir（同一天基準六點有同方向單）、c1_td10_same_day_dir（同一天 C1 或 TD10 同方向）。報三種結果：(a) 單獨；(b) 只在基準沒單的日子；(c) 和基準（Pine 平倉即記帳）先到先做合併，每日 $3,000 檢查照 BRIEF。不要用『基準優先』（RT：事後設計）。",
    "QC9 多重檢定：總組合 76 種，全部報告。通過＝探索、驗證、測試都賺＋有序參數的相鄰值也賺。穩健性（停損緩衝 ±2 點、進場晚 1 分、成本 ×2、進出各 2 點滑價）只對通過者做、只報告不挑。",
    "QC10 每個策略每個組合每天最多 1 筆；多空同分鐘 → 不做；先出現的訊號優先。",
]

S = []
S.append(dict(
    id="S11", name="TURTLE_SOUP_NDAY_v2", family="破底翻／假突破", priority=1,
    timeframe="日線價位（RTH）＋1 分 K 觸發",
    entry_window_et="09:30～12:00（訊號收盤要在窗內；QC3）",
    definitions=[
        "N 日低／高＝前 N 個 RTH 日（09:30～15:59）的最低低點／最高高點；這 N 天必須和當天同一合約（QC7）。",
        "舊極值至少 3 個交易日前形成（Raschke）：N 日低所在那天距今 >= 3 個交易日，否則當天這方向不做。因此 N 日低一定不是 PDL。",
        "做多：09:30 之後第一次 1 分 K 低點 <= N 日低 − 3；從那一分鐘起 30 分鐘內，第一根 1 分 K 收盤 >= N 日低 + 0.5 → 訊號（在那根收盤確定）。30 分鐘內沒收回 → 當天做多作廢。",
        "做空完全相反（高點 >= N 日高 + 3，30 分鐘內收盤 <= N 日高 − 0.5）。",
        "es=on：QC6，ES 用自己的 N 日低（同 N、同合約規則），跌破 0.75 點、收回 +0.25 點，ES 訊號確定時間 <= NQ 訊號確定時間。",
    ],
    entry="訊號那根 1 分 K 收盤後下一分鐘開盤進場（QC3）。當天只做第一個訊號。",
    stop="多：09:30 以來最低低點 − 2 點；空：最高高點 + 2 點。QC2（<15 用 15；>145 不做）。",
    exit="exit=2R：止盈 2R，15:59 時間出場。exit=B55：QC4。",
    params_to_explore=[{"name": "N", "values": [5, 10, 20]}, {"name": "exit", "values": ["2R", "B55"]}, {"name": "es", "values": ["off", "on"]}],
    params_fixed=["跌破 3 點", "收回 +0.5 點", "收回時限 30 分鐘", "舊極值 >= 3 日前", "停損＝當日極端外 2 點"],
    n_combos=12,
    changes=[
        "刪 N=1：就是基準的 AM-PD-k1（前高空／前低多），重測沒有意義。",
        "N 改 {5, 10, 20}：有三個有序值才能檢查高原。",
        "加 30 分鐘收回時限（原版沒有時限，兩小時後收回已不是 soup）。",
        "EOD 改 B55（反轉單沒有抱到收盤的證據；B 證明保本無效；B55 可直接和基準比）。",
        "加 ES 同步參數（f_family 證明對同一機制有效）。",
        "換月：只用同合約，跨換月不做（原規格會抓到舊合約價位）。",
        "停損下限 15、10:00 規則。",
    ],
    overlap="機制同基準的 AM-PD-k1，但價位是 >= 3 日前的 N 日極值，事件不同；同一天常和『前低多／前高空』同方向（價格要先穿過 PDL 才到 N 日低），必須報『只在基準沒單的日子』。和 C1／TD10 方向通常相反（C1 順開盤動能）。",
    rationale="Raschke／Connors Turtle Soup：追 N 日新低的人在假突破時被套，回補造成反彈。先前研究同一機制（前高空）在四段都賺，這裡換成更大的流動性池。",
    critiques="觸發次數有限（未加時限前 N=5 約 217 天、N=20 約 100 天有訊號）；強趨勢日假突破會變真突破；當日極端外 2 點的停損在深跌後常 > 145 點而不做。",
    sources=SRC["S11"],
    expected_1111="不交易（同 P：前 5／10／20 日高低都沒碰到）。",
))
S.append(dict(
    id="S06", name="ICT_SMT_NQ_ES_v2", family="ICT", priority=1,
    timeframe="5 分 K（NQ 與 ES 同時間對齊）",
    entry_window_et="第 k 根 5 分 K 開始時間 >= 09:45；window=AM：確認那根收盤 <= 11:00；DAY：<= 15:00（QC3）",
    definitions=[
        "參考極值：NQ_low_sofar＝09:30 到第 k−1 根的 NQ 最低低點；ES_low_sofar 同理；高點同理。k 從 09:45 那根開始（至少 3 根形成參考）。",
        "多方 breaker=NQ（ICT 原版，逆相對強弱）：L_NQ[k] < NQ_low_sofar 而且 L_ES[k] >= ES_low_sofar。",
        "多方 breaker=ES（順相對強弱）：L_ES[k] < ES_low_sofar 而且 L_NQ[k] >= NQ_low_sofar。",
        "空方用高點，完全相反（breaker=NQ：NQ 破 RTH 高、ES 沒破 → 做空 NQ；breaker=ES：ES 破、NQ 沒破 → 做空 NQ）。",
        "確認：k 之後 3 根 5 分 K 內，NQ 收盤 > H_NQ[k]（多）／< L_NQ[k]（空）。ES 那根缺值 → 不判斷。",
    ],
    entry="確認那根收盤後下一分鐘開盤進場（QC3：落在 09:58～10:02 就 10:03 進場）。",
    stop="多：第 k 根到確認那根的 NQ 最低低點 − 3 點；空相反。QC2。",
    exit="exit=2R：2R，15:59 出場。exit=B55：QC4。",
    params_to_explore=[{"name": "breaker", "values": ["NQ", "ES"]}, {"name": "window", "values": ["AM", "DAY"]}, {"name": "exit", "values": ["2R", "B55"]}],
    params_fixed=["參考極值＝RTH 開盤以來", "k >= 09:45", "確認 3 根", "停損＝極端外 3 點"],
    n_combos=8,
    changes=[
        "滾動 LB 改成『RTH 開盤以來極值』：LB=12 版本 98% 的日子有訊號、第一個訊號中位在 09:35，等於開盤隨機反轉單；拿掉 LB 參數。",
        "把兩種背離拆開（breaker=NQ／ES）：先前研究支持順 NQ 相對強弱（B3、F5b、C 的 10:00 判斷器），兩種預測方向相反，不能混在一起。",
        "加 B55 出場；10:00 規則；停損下限 15。",
    ],
    overlap="breaker=NQ 是『ES 沒同步』，和 f_family 的 ES 同步證據方向相反；breaker=ES 和 C／A 的『NQ 相對 ES』訊號同源，報 C1／TD10 同天同向比例。和基準早盤夜低多 ES 可能同一天反向。",
    rationale="NQ 與 ES 高度相關；在當天極值處一個破、一個不破，代表破的那邊沒被確認，常是掃停損後反轉（ICT SMT）；或代表沒破的那邊相對強（相對強弱）。",
    critiques="趨勢日兩者可以輪流創新低而不反轉；DAY 窗下午訊號少且品質不明；ES 近期 Yahoo 資料要檢查缺值。",
    sources=SRC["S06"],
    expected_1111="09:50～09:54：NQ 低 28,199 < 09:30 以來 NQ 最低 28,205；ES 低 7,596.0 >= ES 09:30 以來最低 7,593.25 → breaker=NQ 多方 SMT；09:55 收 28,291.5 > 28,286.5 確認 → 原本 10:00 進場，依 QC3 改 10:03 開盤（約 28,29x）做多，停損 28,196，2R 約 28,49x（10:50 左右達到，需回測確認）。breaker=ES 當天沒有對應訊號（需實算）。以上用 P 的價位粗算。",
))
S.append(dict(
    id="S03", name="ICT_PO3_JUDAS_v2", family="ICT", priority=1,
    timeframe="5 分 K（收盤確認）；1 分 K（低點／高點）",
    entry_window_et="09:30～11:00（確認那根收盤在窗內；QC3）",
    definitions=[
        "錨點 A：mid＝午夜 00:00～00:14 第一個有效 1 分 K 開盤；0830＝08:30～08:44 第一個有效開盤。找不到 → 當天不做。",
        "多方 Judas：09:30 開盤 O930 > A（先在錨點上方）；09:30 之後到目前為止的最低低點 LL < A（至少 0.25 點）；從 LL < A 那根 5 分 K（含）起，第一根收盤 > A 的 5 分 K ＝確認。",
        "空方：O930 < A；09:30 後最高 HH > A；之後第一根 5 分 K 收盤 < A ＝確認。",
        "O930 == A → 當天不做。多空誰先確認做誰，當天 1 筆。",
    ],
    entry="確認那根 5 分 K 收盤後下一分鐘開盤（QC3）。",
    stop="多：09:30 以來最低低點 LL − 3 點；空：HH + 3 點。QC2。",
    exit="exit=2R：2R、15:59。exit=EOD：不設止盈，浮盈 >= 1R 後停損移到進場價（下一分鐘生效），15:59 出場。exit=B55：QC4。",
    params_to_explore=[{"name": "anchor", "values": ["mid", "0830"]}, {"name": "exit", "values": ["2R", "EOD", "B55"]}],
    params_fixed=["操縱深度 0.25 點", "5 分收盤確認", "停損＝RTH 極端外 3 點", "必須開在交易方向那一側"],
    n_combos=6,
    changes=[
        "刪 anchor=0930：等於 A_trend F6a『收復開盤』（4,320 種，探索／驗證中位每筆 −1／−14）。",
        "加『09:30 開在交易方向那一側』：原規格會把『開在 A 下方後突破 A』也當 Judas，混進相反的突破邏輯。",
        "加 B55 出場；10:00 規則；停損下限 15。",
    ],
    overlap="和基準的掃價位單可能同一天同方向（若 A 附近剛好是 PDL／ONL）；和 C1 常相反（C1 順開盤 5 分鐘方向，Judas 是先逆後順）。",
    rationale="Power of 3：真正的日內方向開始前，先往反方向假動作拿停損；午夜開盤是 ICT 的『真正開盤』。午夜／08:30 開盤這兩個價位在先前研究中沒用過。",
    critiques="強趨勢日開盤就單邊走、不回錨點 → 沒訊號；錨點離開盤很遠時 LL<A 要深跌，停損常 >145 而不做；沒有可信的公開統計。",
    sources=SRC["S03"],
    expected_1111="mid：A＝28,212，O930 28,352.75 > A；09:50～09:54 低 28,199 < A、收 28,246 > A → 確認；09:55 開約 28,247.75 做多，停損 28,196。2R 版約 +103 點、EOD 版約 +236 點（P 粗算）、B55 版 +55 點。0830：A＝28,381.75，O930 < A，09:30～09:34 高 28,426 > A、收 28,348.75 < A → 做空，約 −80 點（同 P）。",
))
S.append(dict(
    id="S02", name="ICT_2022_MODEL_v2（含 OTE）", family="ICT", priority=2,
    timeframe="5 分 K（掃蕩、MSS、FVG）；1 分 K（觸價）",
    entry_window_et="09:30～11:30（掃蕩、MSS、成交都在窗內；QC3）",
    definitions=[
        "流動性池固定 {PDH, PDL, ONH, ONL}（features.daily_levels）。",
        "掃蕩：依 C7，5 分 K，sweep_buf=2，同一根收回（多：L[k] <= 池 − 2 且 C[k] > 池）。同一根掃多個池取最極端。",
        "MSS：k 之後 6 根 5 分 K 內，收盤 > 『第 k 根收盤時已確定的最近 5 分轉折高點（n=2）』（多）；空相反。",
        "FVG（entry=top／ce 用）：5 分 K、C4、min_size=3、disp 不要求、方向同 MSS，第 3 根落在 [k+1, MSS 那根]（必須在 MSS 收盤時已確定），取當時最後形成的一個，之後凍結。沒有 FVG → top／ce 這次不做。",
        "OTE（entry=ote 用）：leg_low＝L[k]（掃蕩低點）、leg_high＝第 k 根到 MSS 那根的最高高點，MSS 收盤時凍結；掛單價＝leg_high − 0.705×(leg_high − leg_low)。空方相反。",
        "es=on：QC6，ES 在同一根或更早也掃了它自己的同名池（PDH/PDL/ONH/ONL，sweep_buf 0.5）並同根收回。",
    ],
    entry="MSS 收盤後開始掛單（QC1 真限價）：top＝FVG 上緣、ce＝FVG 中線、ote＝OTE 價；12 根 5 分 K（60 分鐘）內有效。失效：5 分收盤 < FVG 下緣（top/ce）或 < leg_low（ote）、或超時。",
    stop="掃蕩那根極端外 3 點（多：L[k] − 3）。QC2。",
    exit="target=2R：2R，15:59。target=pool：進場價另一邊最近、自 09:30 起尚未被觸及的池（同 4 個池）；RR<1 → 2R。target=B55：QC4。",
    params_to_explore=[{"name": "entry", "values": ["top", "ce", "ote"]}, {"name": "target", "values": ["2R", "pool", "B55"]}, {"name": "es", "values": ["off", "on"]}],
    params_fixed=["pools=PD+ON", "5 分 K", "sweep_buf=2", "MSS 時限 6 根", "pivot n=2", "FVG min_size=3", "掛單時限 12 根", "OTE=0.705"],
    n_combos=18,
    changes=[
        "併入 S04 OTE（entry=ote），OTE 的波段在 MSS 收盤時凍結，不再隨新高移動。",
        "FVG 必須在 MSS 收盤時已確定（原規格允許 MSS+1，在那根未收盤前結果不唯一）。",
        "pools 固定 PD+ON（ALL 加入的 London 已失敗、PWH/PWL 跨換月、Asia 通常在夜盤區間內）。",
        "C9 近似改真限價（QC1）。",
        "加 B55 出場與 ES 同步。",
    ],
    overlap="掃蕩事件和基準 AM-PD-k1、AM-ON-k1 是同一批（前高空、前低多、夜低多 ES）；A2 ON_SWEEP_2R 驗證期失敗。本策略要回答的問題：同一事件晚一點、在回檔處進場，是否勝過基準在收回時進場。必須報『只在基準沒單的日子』與合併後增量。",
    rationale="先拿流動性、再位移反轉；MSS 證明方向改變，回檔到 FVG／OTE 是低風險進場點。",
    critiques="強趨勢反轉日不回檔、掛單不成交（11/11 就是）；和基準高度重疊，合併增量可能為負。",
    sources=SRC["S02"] + SRC["S04"],
    expected_1111="空方：09:30 那根掃 ONH 28,406.25 收回；MSS 在 09:40 那根。P 算出 5 分 FVG 不成立 → top/ce 不交易。ote：leg_high 28,426（09:30 那根高）、leg_low＝MSS 時最低（約 28,254），掛空價約 28,375；10:25 那根高 28,388.5 觸價做空，停損 28,429，10:40 停損（約 −55 點，需實算）。",
))
S.append(dict(
    id="S10", name="WYCKOFF_SPRING_UPTHRUST_v2", family="破底翻／假突破", priority=2,
    timeframe="5 分 K",
    entry_window_et="10:00～14:30（訊號收盤要在窗內；QC3 → 實際最早 10:03 進場）",
    definitions=[
        "交易區間：第 k 根之前 12 根 5 分 K 的最高 RH、最低 RL；R＝RH−RL，必須 20 <= R <= 0.35×ADR10。",
        "Spring（多）：L[k] < RL − 2 且 C[k] > RL。Upthrust（空）：H[k] > RH + 2 且 C[k] < RH。",
        "confirm=now：第 k 根收盤即訊號。confirm=test：k 之後 6 根內第一根滿足 L > L[k]、V < V[k]、C > O 的 5 分 K，以它收盤為訊號（空方相反）。",
        "vwap=on：QC5，訊號那根收盤時，做多要收盤 > VWAP、做空要 < VWAP（上升趨勢中的盤整掃低才做多）；不符合就看下一個訊號。",
    ],
    entry="訊號收盤後下一分鐘開盤（QC3）。當天第一個合格訊號。",
    stop="多：L[k] − 3；空：H[k] + 3。QC2（<15 用 15）。",
    exit="exit=range：多方目標 RH（區間對側；RR<1 → 2R），15:59。exit=B55：QC4。",
    params_to_explore=[{"name": "confirm", "values": ["now", "test"]}, {"name": "vwap", "values": ["off", "on"]}, {"name": "exit", "values": ["range", "B55"]}],
    params_fixed=["區間 12 根", "R 介於 20 點與 0.35×ADR10", "突破 2 點", "停損極端外 3 點"],
    n_combos=8,
    changes=[
        "加 VWAP 同側參數：盤整條件在 10:00～14:30 有 52% 時間成立，第一個訊號接近隨機，需要方向過濾。",
        "target=2R 換成 B55（range 是 Wyckoff 原意，保留）。",
        "停損下限 15（11/11 範例停損只有 17、28 點）。",
        "10:00 規則。",
    ],
    overlap="和 family 的『掃價位收回』同機制，但價位是滾動 60 分區間，不是固定價位；PM 段和 family PM-IB 可能同一天。報基準同天同向比例。",
    rationale="Wyckoff：區間邊緣被突破立刻收回，代表那一邊沒有後續供需；量縮測試再確認更可靠。",
    critiques="日內 60 分區間只是 Wyckoff 結構的粗略近似；Yahoo 近期量能來源不同，test 版近期結果要註明。",
    sources=SRC["S10"],
    expected_1111="vwap=off、now：12:05 做空 28,610.5（P 的價位）。range 版停損 28,627.5（17 點，依 QC2 放大成 15 點下限以上，維持 17）→ 12:20 停損約 −17 點；B55 版目標 28,555.5，12:05～12:09 低 28,575 沒到，之後最高 28,697 沒碰到 −145 停損（28,755.5），進場 >= 12:00 → 15:49 時間出場（收盤附近約 28,48x，需實算，可能是一筆大賺的空單——這是『固定寬停損抱到時間出場』的效果，不是訊號好）。vwap=on：12:00 價格在 VWAP 上方 → upthrust 做空被擋，只會做 spring 多單（需實算）。",
))
S.append(dict(
    id="S01", name="ICT_SILVER_BULLET_v2", family="ICT", priority=2,
    timeframe="1 分 K（FVG、進出場）；ATR14_1m（當天 slot 0 起算，前 14 根暖身）",
    entry_window_et="AM 10:00～10:59（10:03 前不成交，QC3）；PM 14:00～14:59。FVG 第 3 根要在窗內收盤，成交也要在窗內。",
    definitions=[
        "FVG：C4，1 分，min_size=4，第 2 根實體 >= 1.5×ATR14_1m（ATR 取第 k−2 根收盤時），方向同 FVG。",
        "窗內第一個符合 filter 的 FVG 才算；它在窗內沒成交 → 當窗不交易。",
        "filter=any：不過濾。filter=sweep：多方 FVG 第 1 根往前 10 根內（含）有一根 1 分 K 低點 < 它前 30 根的最低低點；空相反。filter=vwap：FVG 第 k 根收盤 > VWAP 才算多方（空方 < VWAP），QC5。",
    ],
    entry="FVG 確定後掛限價（多：上緣；空：下緣），QC1 真限價，窗內有效。",
    stop="多：FVG 第 1 根低點 − 1；空：第 1 根高點 + 1。QC2（<15 用 15）。",
    exit="exit=2R：2R，時間出場 AM→11:59、PM→15:59。exit=B55：QC4。",
    params_to_explore=[{"name": "window", "values": ["AM", "PM"]}, {"name": "filter", "values": ["any", "sweep", "vwap"]}, {"name": "exit", "values": ["2R", "B55"]}],
    params_fixed=["min_size=4", "disp=1.5×ATR14_1m", "停損 FVG 第 1 根外 1 點", "sweep 回看 30 根、距離 10 根"],
    n_combos=12,
    changes=[
        "刪 LON 窗（倫敦盤相關全部失敗）。",
        "C9 近似改真限價：原規則會刪掉穿過停損的快速虧損單。",
        "加 filter=vwap（盤中順勢）；加 B55 出場。",
        "停損下限 15（1 分 FVG 結構停損常 10～25 點，成本＋滑價佔太多）。",
        "10:00 數據規則。",
    ],
    overlap="AM 窗和 TD10（10:00 進場）同一時段，報同天同向比例；和基準 AM 單可能同時持倉（每日 $3,000 檢查）。",
    rationale="ICT：10–11、14–15 是演算法找流動性後再平衡的時段；FVG 是單邊失衡，回測時有剩餘委託。",
    critiques="1 分 FVG 很多是雜訊（AM 窗約 60% 的日子有合格 FVG）；沒有原作者的停損／目標規則；真限價下成交後立刻打停損的比例可能很高。",
    sources=SRC["S01"],
    expected_1111="AM：P 算出沒有合格 FVG → 不交易（三種 filter 都是）。PM：14:04 空方 FVG，窗內沒回到 28,643.5 → 不交易；vwap filter 下空方 FVG 在 VWAP 上方 → 不合格。",
))
S.append(dict(
    id="S07", name="SMC_TREND_OB_PD_v2", family="SMC", priority=3,
    timeframe="15 分 K（趨勢／區間）；5 分 K（OB）；1 分 K（觸價）",
    entry_window_et="09:45～14:30（QC3）",
    definitions=[
        "15 分趨勢：C8，pivot n=2，從前一交易日 09:30 起連續計算。",
        "交易區間（多頭）：range_low＝最近確定的 15 分轉折低點；range_high＝之後到上一根已收盤 15 分 K 的最高高點；EQ＝中點。",
        "5 分 OB：C5（5 分、n=2 的 BOS 事件），尚未被觸碰、未失效的多方 OB，取最近形成的一個；OB 上緣 <= EQ 才可用（折價區，固定）。",
        "vwap=on：觸價前一分鐘收盤 > VWAP 才做多（QC5）；空方相反。",
        "空頭狀態完全相反。",
    ],
    entry="QC1 真限價掛在 OB 上緣；成交當下（用上一根已收盤 15 分 K）狀態必須仍為多。",
    stop="OB 下緣 − 3（多）。QC2。",
    exit="target=range：range_high（進場時凍結；RR<1 → 2R），15:59。target=B55：QC4。",
    params_to_explore=[{"name": "vwap", "values": ["off", "on"]}, {"name": "target", "values": ["range", "B55"]}],
    params_fixed=["pd=on（折價區）", "15 分 pivot n=2", "5 分 pivot n=2", "OB 定義 C5", "停損 OB 下緣−3"],
    n_combos=4,
    changes=[
        "pd 固定 on（off 只是更多筆的同一件事，不是另一個假說）。",
        "加 vwap 參數（盤中順勢）；target=2R 換成 B55。",
        "C9 近似改真限價；停損下限 15；10:00 規則。",
    ],
    overlap="順勢回檔類，和 A_trend F3（VWAP 回檔，幾乎全賠）同類；保留一個代表。和 C1／TD10 同方向的日子多，報重疊。",
    rationale="SMC：順高時間框結構，在折價區的 OB 買；OB 是機構最後一次反向建倉的位置。",
    critiques="先前回檔掛單類證據不利；V 型反轉日不回測；OB 定義（[m−3, m] 最後一根陰線）任意但客觀。",
    sources=SRC["S07"],
    expected_1111="P 粗算：15 分翻空後 10:25 觸空方 OB 做空、停損 → 約 −30～−80 點；vwap=on 時 10:25 價格在 VWAP（約 28,33x）上方 → 空單被擋（需實算）。",
))
S.append(dict(
    id="S09", name="MSNR_FRESH_LEVEL_v2", family="SNR（Malaysian SnR）", priority=3,
    timeframe="15 分 K（畫水平）；60 分 K（storyline）；5 分 K（reject 確認）；1 分 K（觸價）",
    entry_window_et="09:30～15:00（QC3）；每天 1 筆",
    definitions=[
        "只用收盤價。A-level：15 分收盤轉折高點（C3 用在 C 序列、n=2），價位＝C[k]；V-level：收盤轉折低點。k+2 收盤確定。",
        "Gap level：k−1、k 兩根同色 15 分 K，第 k 根實體 >= 1.0×ATR14_15m（ATR 取 k−1 收盤時）→ 多方 gap＝C[k−1]（支撐）；兩根陰線 → 空方 gap＝C[k−1]（壓力）。k 收盤確定。",
        "只保留前 2 個 RTH 日 09:30 起到當下產生的水平，而且必須和當天同一合約（QC7）。",
        "Fresh：確定後沒有任何 K 棒影線碰到。被碰一次就不再交易。Refresh：15 分收盤穿過水平 → 角色反轉、從那根收盤起重新 fresh。",
        "storyline 固定 on：最近一次 60 分收盤突破的是 A-level（往上）→ 只做支撐；突破 V-level → 只做壓力。",
        "level_type=AV 只用 A/V；GAP 只用 gap level。",
    ],
    entry="entry=limit：QC1 真限價掛在 fresh 支撐（做多）／壓力（做空）。entry=reject：5 分 K 影線碰到 fresh 支撐、收盤 > 該水平 → 下一分鐘開盤做多（壓力相反）；這根 K 碰到後水平就變 unfresh。",
    stop="limit：水平價 − max(15, 0.75×ATR14_15m)；reject：該 5 分 K 低點 − 3。空方相反。QC2。",
    exit="exit=2R：2R，15:59。exit=B55：QC4。",
    params_to_explore=[{"name": "level_type", "values": ["AV", "GAP"]}, {"name": "entry", "values": ["limit", "reject"]}, {"name": "exit", "values": ["2R", "B55"]}],
    params_fixed=["15 分", "收盤 pivot n=2", "gap 實體 1.0×ATR", "回看 2 個 RTH 日", "storyline=on", "每天 1 筆"],
    n_combos=8,
    changes=[
        "加 entry=reject（5 分收盤收回才進場），原版『第一次觸價就進場』＝接刀。",
        "回看 3 日 → 2 日且同合約；storyline 固定 on；target=next 改 B55；每天 2 筆改 1 筆。",
        "C9 近似改真限價；停損下限 15；gap 的 ATR 取哪一根寫清楚；10:00 規則。",
    ],
    overlap="reject 版和 family『掃價位收回』同做法，只是價位換成 MSNR 收盤水平，可以直接比較價位品質；和基準同天同向要報。",
    rationale="Malaysian SnR：用收盤畫的水平才是真正成交的價位；第一次回測反應最大，被碰過的水平委託已消耗。",
    critiques="來源以社群文件為主、沒有獨立統計；storyline 是代用定義；水平多時觸價頻繁，可能接近隨機。",
    sources=SRC["S09"],
    expected_1111="AV＋limit：P 粗算 09:40 觸 V-level 28,266.75 做多、約 −35 點停損。AV＋reject：09:40～09:44 低 28,254 碰到、收 28,288.25 > 28,266.75 → 09:45 做多，停損 28,251（約 −37 點，09:45～09:49 低 28,205 打掉）。GAP 需實算。",
))

DELETED = [
    dict(id="S04", name="ICT_OTE", action="併入 S02（entry=ote）", reason="和 S02／S07／S08 同骨架；leg 隨新高移動、多段並存定義不清；費波那契比例沒有證據；A_trend F3 回檔進場幾乎全賠。"),
    dict(id="S05", name="ICT_NDOG_NWOG", action="刪除", reason="NDOG 大小中位 6.8 點（雜訊）；NWOG 跨換月；『只用最新一個』偏離原意；觸 CE 樣本約 119～166 天再經確認後很小；沒有任何先前證據。"),
    dict(id="S08", name="SMC_CHOCH_IDM", action="刪除", reason="骨架＝S02；IDM 沒有公認定義；條件串過長，預期每年個位數到十幾筆；對 1 根 K 的確認時間差非常敏感。"),
    dict(id="S12", name="ORB_ZARATTINI", action="刪除", reason="T=5＝C1 開盤驅動（已通過、RT 已重算）；T=15/30＝A_trend F2、A1 ORB、ORB30 +55/−145 都已失敗；A1 噪音區間突破弱；停損 >145 不做會刪掉 22%（T=15）～40%（T=30）的寬區間日，正是 ORB 賺錢的趨勢日。"),
    dict(id="S13", name="IB_EXTENSION", action="刪除", reason="A1 60 分 IB 突破 ≈0、45 分以上區間失敗；A_trend F4 新高新低 ≈0；narrow 只占 12% 的日子；唯一新的是出場，B 證明出場優化最易過擬合。"),
    dict(id="S14", name="VWAP_SIGMA_REVERT", action="刪除", reason="方向上違反已證明有效的 VWAP 同側；A2 過度延伸反向 18/20 虧；C 價值區外回歸測試期虧；改成順 VWAP 就是已失敗的 A_trend F3。"),
    dict(id="S15", name="SESSION_RANGE_BREAK", action="刪除", reason="ON-fail＝基準 AM-ON-k1、A2 ON_SWEEP_2R 驗證期轉負；ON-retest＝A1 隔夜高低突破 ≈0；LON 失敗；ASIA 在夜盤區間內部、沒有資訊。有意義的部分已在 S02 v2。"),
    dict(id="S16", name="RTH_GAP_FADE", action="刪除", reason="進場方向＝第一根 5 分 K 方向＝C1 家族子集；A1 跳空 64 種不穩、C 缺口回補測試期虧、P2_quant Q04 重複；gmax 兩點沒有高原可言。"),
]

order = ["S11", "S06", "S03", "S02", "S10", "S01", "S07", "S09"]
total = sum(s["n_combos"] for s in S)
assert total == 76, total
for s in S:
    k = 1
    for p in s["params_to_explore"]:
        k *= len(p["values"])
    assert k == s["n_combos"], (s["id"], k)
    assert len(s["params_to_explore"]) <= 3
out = dict(
    author="Q1_critic",
    based_on="agents/P_strategist/strategies.json（16 個策略，90 種組合）",
    note="研究用途，不是投資建議。修正版只放要回測的 8 個策略；審查理由見 critique.txt。規則仍照 BRIEF（下一分鐘開盤進場、1 口 NQ、每筆扣 $48、同分鐘算止損、每日 $3,000、15:59 平倉）。",
    common_overrides=COMMON,
    total_param_combos=total,
    priority_order=order,
    pass_criteria_extra=[
        "BRIEF 的通過標準（探索、驗證、測試都賺＋相鄰參數也賺）。",
        "同一訊號集合的 2R（或 range／pool／EOD）與 B55 至少一個通過；只有前者通過時註明『靠出場』。",
        "『只在基準沒單的日子』也要賺。",
        "和基準先到先做合併（Pine 記帳）後，測試期合計不少於基準單獨。",
    ],
    strategies=S,
    deleted=DELETED,
)
json.dump(out, open(os.path.join(HERE, "strategies_v2.json"), "w"), ensure_ascii=False, indent=1)
print("written", total, [s["id"] for s in S])
