"""不偷看未來的自動檢查（截斷測試）。研究用途，不是投資建議。
對每個訊號：把當天（NQ 與 ES）進場那一分鐘起的資料全部改成 NaN（限價單：成交分鐘只留 H/L 當觸價依據，O/C/V 改 NaN，
之後全部 NaN），只對那一天重跑產生器，訊號（方向、進場/成交 slot、價位、停損、目標）必須完全相同。
另外每個策略隨機抽 5 筆印出進場前用到的數值與時間。"""
import sys, json, random
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import r1lib as R
import strat as S
from run import sig_sets

random.seed(11)
ARR = [R.O, R.H, R.L, R.C, R.V, R.EO, R.EH, R.EL, R.EC, R.EV]
TRAD0 = R.TRAD.copy(); ETRAD0 = R.ETRAD.copy()


def rerun(fn, i, cut, keep_hl):
    saved = [a[i].copy() for a in ARR]
    try:
        for k, a in enumerate(ARR):
            if keep_hl and k in (1, 2, 6, 7):          # H、L、EH、EL 在成交分鐘保留
                a[i, cut + 1:] = np.nan
            else:
                a[i, cut:] = np.nan
        R.TRAD[:] = False; R.TRAD[i] = TRAD0[i]
        R._VW.clear()
        out = fn()
    finally:
        for a, s in zip(ARR, saved):
            a[i] = s
        R.TRAD[:] = TRAD0; R._VW.clear()
    return [z for z in out if z["i"] == i]


def key(z):
    return (z["d"], z["kind"], z.get("e0"), z.get("e"), round(float(z.get("P", 0) or 0), 4), round(float(z["stop_raw"]), 4),
            None if z.get("tgt") is None else round(float(z["tgt"]), 4))


lines = []
def P(s=""):
    print(s); lines.append(str(s))


SIGS = pd.read_pickle(R.OUT + "/all_sigs.pkl")
bad_total = 0
for sid in ["S11", "S06", "S03", "S02", "S10", "S01", "S07", "S09"]:
    n_chk = n_bad = 0
    sets = sig_sets(sid)
    samples = []
    for sp, fn in sets:
        sigs = SIGS[(sid, tuple(sp.items()))]
        for z in sigs:
            cut = z["e0"] if z["kind"] == "mkt" else z["e"]
            got = rerun(fn, z["i"], cut, z["kind"] == "lim")
            ok = len(got) == 1 and key(got[0]) == key(z)
            n_chk += 1
            if not ok:
                n_bad += 1
                if n_bad <= 3:
                    P(f"  !! 不一致 {sid} {sp} {R.DATES[z['i']]} 原 {key(z)} 截斷後 {[key(g) for g in got]}")
            samples.append((sp, z))
    bad_total += n_bad
    P(f"{sid}: 截斷測試 {n_chk} 個訊號，不一致 {n_bad}")
    for sp, z in random.sample(samples, min(5, len(samples))):
        e = z["e0"] if z["kind"] == "mkt" else z["e"]
        P(f"   樣本 {R.DATES[z['i']]} {sp} 方向{'多' if z['d'] == 1 else '空'} {z['kind']} 訊號/觸價分鐘 {R.hhmm(z['sig'])} "
          f"進場/成交 slot {R.hhmm(e)} 用到資料最晚 {R.hhmm(z['known'])} 停損(QC2前) {z['stop_raw']:.2f} "
          f"{'掛單 %.2f' % z['P'] if z['kind'] == 'lim' else ''} 目標 {z.get('tgt')}")
        P("      " + json.dumps({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v if not isinstance(v, tuple) else [float(x) for x in v])
                                 for k, v in z["info"].items()}, ensure_ascii=False))
        assert z["known"] < e
P(f"合計不一致：{bad_total}")
open(R.OUT + "/lookahead.txt", "w").write("\n".join(lines))
