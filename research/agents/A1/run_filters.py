"""ORB filter study (discovery only)."""
import warnings; warnings.filterwarnings("ignore")
from a1lib import *
from signals import *
g, lv = load(); d = daytable(g, lv); DISC = (g.split == "discovery")
D = d.set_index("i")
rows = []
def go(name, tr):
    s = summ(tr, g, "discovery"); s["name"] = name
    if len(tr):
        s["L_avg"] = round(tr[tr.dir == 1].pnl.mean(), 0); s["S_avg"] = round(tr[tr.dir == -1].pnl.mean(), 0)
        h = halves(tr); s["h"] = "/".join(str(int(x)) for x in h["mean"].values)
    rows.append(s)
bases = {"ORB15 or_eod": dict(N=15, cutoff="12:00", exit_mode="or_eod"),
         "ORB15 r_k2": dict(N=15, cutoff="12:00", exit_mode="r_k", k=2),
         "ORB30 or_eod": dict(N=30, cutoff="12:00", exit_mode="or_eod"),
         "ORB30 r_k1": dict(N=30, cutoff="12:00", exit_mode="r_k", k=1),
         "ORB60 or_eod": dict(N=60, cutoff="13:00", exit_mode="or_eod")}
for bn, kw in bases.items():
    tr = sim.run_signals(g, orb_break(g, lv, days=DISC, **kw))
    tr = tr.merge(D[["atr14", "open_0930", "pdc", "vah", "val", "pdh", "pdl", "onh", "onl", "i2_dir"]], left_on="i", right_index=True)
    tr["orw"] = tr["orr"] / tr["atr14"]
    q = tr.orw.quantile([1/3, 2/3]).values
    go(bn, tr)
    go(bn + " | OR narrow (<q33)", tr[tr.orw < q[0]])
    go(bn + " | OR mid", tr[(tr.orw >= q[0]) & (tr.orw < q[1])])
    go(bn + " | OR wide (>q67)", tr[tr.orw >= q[1]])
    gapdir = np.sign(tr.open_0930 - tr.pdc)
    go(bn + " | with gap dir", tr[tr.dir == gapdir])
    go(bn + " | against gap", tr[tr.dir == -gapdir])
    outVA = np.where(tr.open_0930 > tr.vah, 1, np.where(tr.open_0930 < tr.val, -1, 0))
    go(bn + " | open in VA", tr[outVA == 0])
    go(bn + " | open outVA, brk same side", tr[(outVA != 0) & (tr.dir == outVA)])
    go(bn + " | open outVA, brk back toward VA", tr[(outVA != 0) & (tr.dir == -outVA)])
    outPR = np.where(tr.open_0930 > tr.pdh, 1, np.where(tr.open_0930 < tr.pdl, -1, 0))
    go(bn + " | open in prior range", tr[outPR == 0])
    go(bn + " | open out prior range", tr[outPR != 0])
    hiv = tr.atr14 > tr.atr14.median()
    go(bn + " | ATR high", tr[hiv]); go(bn + " | ATR low", tr[~hiv])
df = pd.DataFrame(rows).set_index("name")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(df[["n","avg","total","maxdd","ret/dd","sharpe_ann","t","L_avg","S_avg","h"]].to_string())
log_variants("orb_filters", df.reset_index()[["name","n","avg","sharpe_ann"]].to_dict("records"))
