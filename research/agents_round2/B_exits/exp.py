"""跑所有出場／加碼／部位／停損變化，寫 results.csv。研究用途，不是投資建議。
用法：python exp.py   （約數分鐘）"""
import warnings; warnings.filterwarnings("ignore")
import itertools, json
from core import *

ROWS = []


def run(fam, name, p=None, ext=False, atr=False, size_mode=None, size_kw=None, book_kw=None, R=None, size=None):
    if R is None:
        sc = (CAND.atr.values / ATR_REF) if atr else None
        R = simulate(p, ext=ext, scale=sc)
    if size is None:
        size = sizes(R, size_mode, **(size_kw or {}))
    B = book(R, size=size, **(book_kw or {}))
    st = stats(B)
    row = flat(st, name); row["fam"] = fam
    row["avg_c"] = float(np.mean(B.c)) if len(B) else 0
    ROWS.append(row)
    return B, st


INF = np.inf
if __name__ == "__main__":
    base_B, base = run("0基準", "基準 +55/−145 到 12:29（亞洲 02:59）", plan())
    run("0基準", "基準（books17 記帳：預定出場才算已實現）", plan(), book_kw=dict(sched=True))

    # ---- A 固定點數網格 ----
    for tp, sl in itertools.product([30, 40, 55, 70, 85, 100, 125, 150, INF], [60, 80, 100, 125, 145, 175, 200]):
        run("A固定", f"TP{tp}/SL{sl}", plan(sl=sl, tp1=tp))
    for tp, sl in itertools.product([55, 85, 125, INF], [100, 145, 200]):
        run("A固定抱收盤", f"TP{tp}/SL{sl} 抱到 15:59/08:29", plan(sl=sl, tp1=tp), ext=True)
    # ---- A2 風險報酬比 ----
    for sl in [30, 40, 55, 75, 100, 145]:
        for rr in [1, 2]:
            run("A風報比", f"1:{rr} SL{sl}/TP{sl*rr}", plan(sl=sl, tp1=sl * rr))
            run("A風報比", f"1:{rr} SL{sl}/TP{sl*rr} 抱收盤", plan(sl=sl, tp1=sl * rr), ext=True)
    # ---- B ATR 倍數（距離 = 點數 × ATR/ATR_REF；ATR_REF≈258，所以 145 點 ≈ 0.56 ATR）----
    for tpk, slk in itertools.product([0.12, 0.16, 0.21, 0.27, 0.33, 0.45], [0.3, 0.4, 0.5, 0.56, 0.7, 0.85]):
        tp, sl = tpk * ATR_REF, slk * ATR_REF
        run("B_ATR 1口", f"ATR TP{tpk}/SL{slk}", plan(sl=sl, tp1=tp), atr=True)
        run("B_ATR 定風險", f"ATR TP{tpk}/SL{slk} 定風險≤1口", plan(sl=sl, tp1=tp), atr=True, size_mode="risk", size_kw=dict(cap=1.0))
        run("B_ATR 定風險2", f"ATR TP{tpk}/SL{slk} 定風險≤2口", plan(sl=sl, tp1=tp), atr=True, size_mode="risk", size_kw=dict(cap=2.0))
    # ---- C1 保本停損 ----
    for bt, bo in itertools.product([20, 30, 40, 50], [0, 5, 15]):
        run("C保本", f"浮盈{bt}後停損移到 +{bo}", plan(be_trig=bt, be_off=bo))
    # ---- C2 移動停損（不設止盈或遠止盈）----
    for tp, trt, trd, ext in itertools.product([INF, 110], [30, 55, 80], [25, 40, 60, 80], [False, True]):
        run("C移動", f"移動 觸發{trt}/距離{trd} TP{tp}{' 抱收盤' if ext else ''}", plan(tp1=tp, tr_trig=trt, tr_dist=trd), ext=ext)
    # ---- C3 分批：一半 tp1 出，另一半 ----
    for tp1, rest, ext in itertools.product([30, 40, 55, 80], ["抱", "保本抱", "保本移40", "保本移60", "保本TP110", "保本TP150"], [False, True]):
        kw = dict(tp1=tp1, f1=0.5)
        if "保本" in rest: kw["be_after_tp1"] = 1
        if "移40" in rest: kw.update(tr_trig=tp1, tr_dist=40)
        if "移60" in rest: kw.update(tr_trig=tp1, tr_dist=60)
        if "TP110" in rest: kw["tp2"] = 110
        if "TP150" in rest: kw["tp2"] = 150
        run("C分批", f"一半+{tp1}，另一半{rest}{' 到收盤' if ext else ''}", plan(**kw), ext=ext)
    # ---- C4 時間停損 ----
    for tm, mfe in itertools.product([15, 30, 45, 60, 90, 120], [10, 20, 30]):
        run("C時間", f"{tm} 分鐘內浮盈沒到 {mfe} 就出", plan(ts_min=tm, ts_mfe=mfe))
    # ---- D 逆勢攤平（DCA）----
    # D1：1 口進場，−X 再加 s 口，總停損在原進場 −SL，止盈從均價 +TP
    for x, s, sl, tp in itertools.product([30, 50, 75, 100], [0.5, 1.0], [145, 200], [55, 30]):
        if x >= sl: continue
        run("D攤平1", f"1口，−{x} 加 {s} 口，停損 −{sl}，均價+{tp}", plan(sl=sl, tp1=tp, a1d=-x, a1s=s, tp_avg=1))
    # D2：半口進場，−X 再加半口（總 1 口），停損 −SL，均價 +TP（最壞虧損接近基準）
    for x, sl, tp in itertools.product([40, 60, 80, 100, 120], [145, 175, 200, 240], [30, 55]):
        if x >= sl: continue
        p = plan(sl=sl, tp1=tp, a1d=-x, a1s=1.0, tp_avg=1)
        R = simulate(p); run("D攤平半半", f"0.5+0.5，−{x} 加，停損 −{sl}，均價+{tp}", R=R, size=np.full(len(R), 0.5))
    # D3：三段 0.4/0.3/0.3（比例 1:0.75:0.75）
    for x1, x2, sl, tp in itertools.product([40, 60], [90, 120], [160, 200], [30, 55]):
        p = plan(sl=sl, tp1=tp, a1d=-x1, a1s=0.75, a2d=-x2, a2s=0.75, tp_avg=1)
        R = simulate(p); run("D三段攤平", f"0.4/0.3/0.3 在 0/−{x1}/−{x2}，停損 −{sl}，均價+{tp}", R=R, size=np.full(len(R), 0.4))
    # D4：攤平後只求回到原進場價就全出（救援式）
    for x, sl in itertools.product([50, 75, 100], [145, 200]):
        p = plan(sl=sl, tp1=55, a1d=-x, a1s=1.0, tp_after_add=0.0)
        R = simulate(p); run("D救援", f"0.5+0.5，−{x} 加，加後回原價全出，停損 −{sl}", R=R, size=np.full(len(R), 0.5))
    # ---- E 順勢加碼 ----
    for x, s, po, tp in itertools.product([20, 30, 40], [0.5, 1.0], [0, 5], [55, 80, 110]):
        if tp <= x: continue
        run("E順勢加碼", f"1口，+{x} 加 {s} 口，停損移到 +{po}，TP 原價+{tp}", plan(tp1=tp, a1d=x, a1s=s, pyr_off=po))
    for x, po, tp in itertools.product([20, 30, 40], [0, 5], [55, 80, 110]):
        if tp <= x: continue
        p = plan(tp1=tp, a1d=x, a1s=1.0, pyr_off=po)
        R = simulate(p); run("E順勢半半", f"0.5+0.5，+{x} 加，停損移到 +{po}，TP 原價+{tp}", R=R, size=np.full(len(R), 0.5))
    # ---- F 部位大小（在基準出場上）----
    R0 = simulate(plan())
    for lo, hi in [(0.5, 1.5), (0.5, 1.0), (0.3, 1.0), (0.7, 1.3)]:
        run("F波動口數", f"口數＝ATR_REF/ATR 夾在 [{lo},{hi}]", R=R0, size=sizes(R0, "vol", lo=lo, hi=hi))
    # 依近期績效
    def mk_after_loss(k, cut):
        def f(hist, c):
            return c * cut if len(hist) >= 1 and any(h < 0 for h in hist[-k:]) else c
        return f
    def mk_eq(n, cut):
        def f(hist, c):
            return c * cut if len(hist) >= n and sum(hist[-n:]) < 0 else c
        return f
    def mk_win_up(k, up):
        def f(hist, c):
            return c * up if len(hist) >= k and all(h > 0 for h in hist[-k:]) else c
        return f
    for k, cut in itertools.product([1, 2, 3], [0.5, 0.0]):
        run("F績效口數", f"最近 {k} 筆有虧 → 口數×{cut}", R=R0, book_kw=dict(perf=mk_after_loss(k, cut)))
    for n, cut in itertools.product([5, 10, 20], [0.5, 0.0]):
        run("F績效口數", f"最近 {n} 筆合計<0 → 口數×{cut}", R=R0, book_kw=dict(perf=mk_eq(n, cut)))
    for k in [2, 3, 5]:
        run("F績效口數", f"連贏 {k} 筆後 → 1.5 口", R=R0, book_kw=dict(perf=mk_win_up(k, 1.5)))
    # Kelly：用探索期各點的勝率與賠率算 f*，口數 = f*/平均 f* × frac（夾 0.3~1.5）
    D0 = R0[R0.seg == "探索"]
    kel = {}
    for pt, x in D0.groupby("pt"):
        w = x.pnl1[x.pnl1 > 0]; l = -x.pnl1[x.pnl1 <= 0]
        p_ = len(w) / len(x); b = w.mean() / l.mean() if len(l) else 9
        kel[pt] = max(p_ - (1 - p_) / b, 0)
    km = np.mean([kel[p] for p in R0.pt])
    json.dump({k: round(v, 3) for k, v in kel.items()}, open("kelly_disc.json", "w"), ensure_ascii=False)
    for frac in [0.5, 0.75, 1.0]:
        c = np.clip(np.array([kel[p] for p in R0.pt]) / km * frac, 0.3, 1.5)
        run("F_Kelly", f"Kelly 比例分配 ×{frac}（夾 0.3~1.5）", R=R0, size=c)
    # ---- G 每日／每週停損、連虧暫停 ----
    for ds in [1000, 1500, 2000, 2500]:
        run("G日停損", f"當天已實現 ≤ −{ds} 停止", R=R0, book_kw=dict(day_stop=ds))
    for ws in [3000, 4500, 6000, 9000]:
        run("G週停損", f"本週已實現 ≤ −{ws} 停到下週", R=R0, book_kw=dict(week_stop=ws))
    for n, d in itertools.product([2, 3], [1, 2, 3, 5]):
        run("G連虧暫停", f"連虧 {n} 筆暫停 {d} 天", R=R0, book_kw=dict(pause_after=n, pause_days=d))
    for lim in [2000]:
        run("G日上限", f"每日上限改 ${lim}", R=R0, book_kw=dict(lim=lim))

    Rr = pd.DataFrame(ROWS)
    Rr.to_csv("results.csv", index=False)
    print(len(Rr), "rows")
