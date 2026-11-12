"""第二批：同風險比較。每筆口數 = 把「最壞虧損」調到和基準一樣（≈$2,948，最多 3 口，0.1 口為單位），
這樣小停損可以多下、攤平可以放大，公平比較結構本身。寫 results2.csv。研究用途，不是投資建議。"""
import warnings; warnings.filterwarnings("ignore")
import itertools
import exp
from core import *

INF = np.inf
exp.run("0基準", "基準 +55/−145", plan())
exp.run("0基準", "基準（books17 記帳）", plan(), book_kw=dict(sched=True))
RK = dict(size_mode="risk", size_kw=dict(cap=3.0))
# A 同風險：停損小 → 口數大
for sl, tp in itertools.product([50, 60, 75, 90, 100, 125, 145, 175], [30, 40, 55, 70, 85, 100]):
    exp.run("A同風險", f"SL{sl}/TP{tp} 同風險", plan(sl=sl, tp1=tp), **RK)
# D 同風險攤平：第一口 c，−x 再加 r×c，總停損 −sl，均價 +tp
for x, r, sl, tp in itertools.product([30, 50, 75, 100, 125], [0.5, 1.0, 2.0], [125, 145, 175, 200, 250], [30, 55]):
    if x >= sl - 20: continue
    exp.run("D同風險攤平", f"−{x} 加 {r}倍，停損 −{sl}，均價+{tp} 同風險", plan(sl=sl, tp1=tp, a1d=-x, a1s=r, tp_avg=1), **RK)
# D 同風險三段
for x1, x2, sl, tp in itertools.product([40, 60], [90, 120], [160, 200], [30, 55]):
    exp.run("D同風險三段", f"1/0.75/0.75 在 0/−{x1}/−{x2}，停損 −{sl}，均價+{tp} 同風險",
            plan(sl=sl, tp1=tp, a1d=-x1, a1s=0.75, a2d=-x2, a2s=0.75, tp_avg=1), **RK)
# E 同風險順勢加碼
for x, r, po, tp in itertools.product([20, 30, 40], [0.5, 1.0], [-30, 0, 5], [55, 80, 110]):
    if tp <= x: continue
    exp.run("E同風險加碼", f"+{x} 加 {r}倍，停損移到 {po:+d}，TP 原價+{tp} 同風險", plan(tp1=tp, a1d=x, a1s=r, pyr_off=po), **RK)
pd.DataFrame(exp.ROWS).to_csv("results2.csv", index=False)
print(len(exp.ROWS))
