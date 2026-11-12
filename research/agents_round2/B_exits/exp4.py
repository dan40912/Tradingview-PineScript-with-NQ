"""第四批：剩餘額度補小口數、依進場點不同止盈、寬停損組合。寫 results4.csv。研究用途，不是投資建議。"""
import warnings; warnings.filterwarnings("ignore")
import itertools
import exp
from core import *
exp.run("0基準", "基準 +55/−145", plan())
exp.run("0基準", "基準（books17 記帳）", plan(), book_kw=dict(sched=True))
R0 = simulate(plan())
exp.run("F餘額", "基準＋額度不夠時用剩餘額度下小口", R=R0, book_kw=dict(fill=True))
for sl in [175, 200, 225]:
    p = plan(sl=sl, tp1=55); R = simulate(p)
    exp.run("F餘額", f"SL{sl}/TP55 同風險＋剩餘額度", R=R, size=sizes(R, "risk", cap=3.0), book_kw=dict(fill=True))
# 依進場點不同止盈（只 2 種事先想好的分法）
def mixed(tp_map, sl=145):
    parts = []
    for tp in set(tp_map.values()):
        R = simulate(plan(sl=sl, tp1=tp)); parts.append(R[R.pt.map(tp_map) == tp])
    return pd.concat(parts).sort_index()
for name, m in [("I2 用 TP85，其他 TP55", {"I2": 85}), ("美盤點 TP85，亞洲點 TP55", {"I2": 85, "前高空": 85, "前低多": 85, "早盤夜低多ES": 85})]:
    tm = {p: m.get(p, 55) for p in CAND.pt.unique()}
    exp.run("H分點止盈", name, R=mixed(tm))
# 寬停損 SL200/TP55（0.7 口）再加一個管理
base200 = dict(sl=200, tp1=55)
for nm, kw in [("＋60分鐘沒到+10出", dict(ts_min=60, ts_mfe=10)), ("＋浮盈50後保本+5", dict(be_trig=50, be_off=5)),
               ("＋一半+55、一半保本 TP110", dict(f1=0.5, be_after_tp1=1, tp2=110))]:
    p = plan(**base200, **kw); R = simulate(p)
    exp.run("I寬停損組合", "SL200/TP55 同風險" + nm, R=R, size=sizes(R, "risk", cap=3.0))
R = simulate(plan(**base200))
exp.run("I寬停損組合", "SL200/TP55 0.7口×波動（夾0.5~1）", R=R, size=0.7 * np.clip(ATR_REF / R.atr.values, 0.5, 1.0))
pd.DataFrame(exp.ROWS).to_csv("results4.csv", index=False)
