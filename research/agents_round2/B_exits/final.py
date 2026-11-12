"""最後的表：四段＋年度、變化數統計、近期逐筆、11/11 示範。輸出 final_tables.md。研究用途，不是投資建議。"""
import warnings; warnings.filterwarnings("ignore")
from core import *
from lib import sim as S

out = []
P = lambda s="": out.append(s)
R0 = simulate(plan())
R200 = simulate(plan(sl=200, tp1=55))
lowrisk = lambda R: np.full(len(R), 0.7)
# 分點止盈
def mixed(tp_map):
    parts = []
    for tp in set(tp_map.values()):
        R = simulate(plan(tp1=tp)); parts.append(R[R.pt.map(tp_map) == tp])
    return pd.concat(parts).sort_index()
Ri2 = mixed({p: (85 if p == "I2" else 55) for p in CAND.pt.unique()})
Rdca = simulate(plan(sl=200, tp1=55, a1d=-30, a1s=0.5, tp_avg=1))
Rtr = simulate(plan(tp1=110, tr_trig=80, tr_dist=25))
Rtp100 = simulate(plan(tp1=100))
Rts = simulate(plan(ts_min=60, ts_mfe=10))
Rpart = simulate(plan(tp1=55, f1=0.5, be_after_tp1=1, tp2=110))
Rbe = simulate(plan(be_trig=40, be_off=5))
cases = [
    ("基準（books17 記帳，官方 445 筆）", R0, None, dict(sched=True)),
    ("基準（實際出場記帳＝Pine 做法）", R0, None, {}),
    ("★低風險版 SL200/TP55 × 0.7 口", R200, lowrisk(R200), {}),
    ("低風險版（books17 記帳）", R200, lowrisk(R200), dict(sched=True)),
    ("TP100/SL145（探索+驗證入選，測試失敗）", Rtp100, None, {}),
    ("I2 用 TP85（探索+驗證入選，測試失敗）", Ri2, None, {}),
    ("移動停損 觸發80/距離25、TP110", Rtr, None, {}),
    ("一半+55、一半保本 TP110", Rpart, None, {}),
    ("浮盈40後保本+5", Rbe, None, {}),
    ("60 分鐘沒到 +10 就出", Rts, None, {}),
    ("攤平：0.5 口，−30 加 0.25 口，總停損 −200，均價+55", Rdca, np.full(len(Rdca), 0.5), {}),
    ("波動口數 ATR_REF/ATR 夾 0.5~1", R0, sizes(R0, "vol", lo=0.5, hi=1.0), {}),
    ("連虧 2 筆暫停 1 天", R0, None, dict(pause_after=2, pause_days=1)),
    ("週停損 −4,500", R0, None, dict(week_stop=4500)),
]
books = {}
P("## 四段績效（和基準並排）\n")
rows = []
for name, R, sz, kw in cases:
    B = book(R, size=sz, **kw); books[name] = B
    rows.append((name, stats(B)))
P(table(rows))
P("\n## 年度（對照 BRIEF 的基準表）\n")
P("| 方案 | 筆數 | 勝率 | 合計 | 最大回撤 | 最差一天 | 報酬/回撤 | 2024 | 2025 | 2026-1~9 | 近期 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name, B in books.items():
    y = np.select([B.date >= "2026-10-01", B.date >= "2026-01-01", B.date >= "2025-01-01"], ["近期", "2026", "2025"], "2024")
    s = B.groupby(y).pnl.sum()
    m = dd(B.pnl.values)
    P(f"| {name} | {len(B)} | {100*(B.pnl>0).mean():.1f}% | {B.pnl.sum():+,.0f} | {m:+,.0f} | {B.groupby('date').pnl.sum().min():+,.0f} | {B.pnl.sum()/-m:.2f} | "
      + " | ".join(f"{s.get(k,0):+,.0f}" for k in ["2024", "2025", "2026", "近期"]) + " |")

# 每口（不受每日 $3,000 限制）比較：停損 145 vs 200
P("\n## 每 1 口、不經每日 $3,000 檢查的原始比較（599 個候選點）\n")
P("| 段 | SL145/TP55 合計 | SL200/TP55 合計 | 差 |\n|---|---:|---:|---:|")
for sg in SEGS:
    a, b = R0[R0.seg == sg].pnl1.sum(), R200[R200.seg == sg].pnl1.sum()
    P(f"| {sg} | {a:+,.0f} | {b:+,.0f} | {b-a:+,.0f} |")

# 近期逐筆
P("\n## 近期（2026-10-01～11-11）逐筆：基準 vs 低風險版\n")
A = books["基準（實際出場記帳＝Pine 做法）"]; Bv = books["★低風險版 SL200/TP55 × 0.7 口"]
A = A[A.seg == "近期"]; Bv = Bv[Bv.seg == "近期"]
why = {0: "時間", 1: "停損", 2: "止盈", 3: "時間停損"}
M = A[["date", "pt", "dir", "entry_slot", "pnl", "why"]].merge(Bv[["date", "pt", "entry_slot", "pnl", "why"]], on=["date", "pt", "entry_slot"], how="outer", suffixes=("_基準", "_低風險"))
P("| 日期 | 點 | 進場 | 基準 | 低風險 0.7 口 |\n|---|---|---|---:|---:|")
for r in M.sort_values(["date", "entry_slot"]).itertuples():
    f = lambda p, w: "—" if pd.isna(p) else f"{p:+,.0f}（{why[int(w)]}）"
    P(f"| {r.date} | {r.pt} | {S.hhmm_of(int(r.entry_slot))} | {f(r.pnl_基準, r.why_基準)} | {f(r.pnl_低風險, r.why_低風險)} |")

# 11/11：示範（這兩個點不在組合內）
P("\n## 2026-11-11 示範：當天唯一的美盤訊號 AM-ON-k1 空（09:35，不在六個點內）若用各種出場\n")
F = pd.read_csv(os.path.join(RESEARCH, "wr_table.csv.gz"))
demo = F[(F.date == "2026-11-11") & (F.rule == "AM-ON-k1")].head(1)
dc = demo.assign(i=demo.date.map(g.dindex), pt="示範", key="AM-ON-k1|空", xs_ext=EXT_AM, atr=ATR["2026-11-11"], seg="近期", shock=0)
for nm, p, c in [("基準 +55/−145", plan(), 1.0), ("低風險 +55/−200", plan(sl=200, tp1=55), 0.7), ("TP100/SL145", plan(tp1=100), 1.0),
                 ("移動 觸發80/距離25 TP110", plan(tp1=110, tr_trig=80, tr_dist=25), 1.0), ("一半+55、一半保本 TP110", plan(f1=0.5, be_after_tp1=1, tp2=110), 1.0)]:
    r = simulate(p, cand=dc.reset_index(drop=True)).iloc[0]
    i = g.dindex["2026-11-11"]; px = g.O[i][int(dc.entry_slot.iloc[0])]
    P(f"- {nm}：進場 {px:,.2f}（{S.hhmm_of(int(dc.entry_slot.iloc[0]))}），{why[int(r.why)]}出場 {S.hhmm_of(int(r.xslot))}，{c} 口損益 {r.pnl1*c:+,.0f}")
open("final_tables.md", "w").write("\n".join(out))
print("\n".join(out))
