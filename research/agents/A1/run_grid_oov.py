"""Out-of-value ORB grid (discovery only)."""
import warnings; warnings.filterwarnings("ignore")
from a1lib import *
from signals import *
g, lv = load(); d = daytable(g, lv); DISC = (g.split == "discovery")
L = lv.set_index("i")
outVA = ((L.open_0930 > L.vah) | (L.open_0930 < L.val)) & L.has_prev
outPR = ((L.open_0930 > L.pdh) | (L.open_0930 < L.pdl)) & L.has_prev
inVA = (~outVA) & L.has_prev
regs = {"outVA": outVA.values, "outPR": outPR.values, "inVA": inVA.values}
rows = []
for reg, m in regs.items():
    days = DISC & m
    for N in [10, 15, 20, 30, 45]:
        for cut in ["11:00", "12:00", "13:00"]:
            for em, k in [("or_eod", None), ("r_k", 1.0), ("r_k", 1.5), ("r_k", 2.0), ("r_k", 3.0)]:
                for fade in ([False, True] if reg == "inVA" else [False]):
                    kw = dict(N=N, cutoff=cut, exit_mode=em, k=k or 2.0, fade=fade, days=days)
                    if fade and em == "or_eod": kw["exit_mode"] = "target_other"
                    tr = sim.run_signals(g, orb_break(g, lv, **kw))
                    s = summ(tr, g, "discovery")
                    if len(tr):
                        h = halves(tr); s["h"] = "/".join(str(int(x)) for x in h["mean"].values)
                        s["hpos"] = int((h["mean"].iloc[:3] > 0).sum())
                    s.update(reg=reg, N=N, cut=cut, exit=(kw["exit_mode"] + (str(k) if em == "r_k" else "")), fade=fade)
                    rows.append(s)
df = pd.DataFrame(rows)
df.to_csv("grid_oov_discovery.csv", index=False)
log_variants("orb_regime_grid", [{"name": f"{r.reg} N{r.N} {r.cut} {r.exit} fade{r.fade}", "n": r.n, "avg": r.avg, "sharpe_ann": r.sharpe_ann} for r in df.itertuples()])
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
cols = ["reg","fade","N","cut","exit","n","avg","total","maxdd","ret/dd","sharpe_ann","t","h","hpos"]
for reg in ["outVA", "outPR", "inVA"]:
    x = df[df.reg == reg]
    print(x[cols].to_string(index=False))
    print(reg, "share avg>0:", round((x.avg > 0).mean(), 2), "share hpos==3:", round((x.hpos == 3).mean(), 2))
