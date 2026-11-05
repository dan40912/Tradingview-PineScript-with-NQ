import warnings, itertools; warnings.filterwarnings("ignore")
from a1lib import *
from signals import *
g, lv = load(); d = daytable(g, lv)
DISC = (g.split == "discovery")
rows = []
def go(name, sigs):
    tr = sim.run_signals(g, sigs)
    s = summ(tr, g, "discovery"); s["name"] = name
    if len(tr):
        s["L_avg"] = round(tr[tr.dir == 1].pnl.mean(), 0); s["S_avg"] = round(tr[tr.dir == -1].pnl.mean(), 0)
    rows.append(s); return tr
for N in [5, 15, 30, 60]:
    for cut in ["11:00", "13:00"]:
        for em in ["or_eod", "half_eod", "r_k1", "r_k2", "r_k3", "fixed55"]:
            kw = {}
            if em.startswith("r_k"): kw = dict(exit_mode="r_k", k=float(em[-1]))
            elif em == "fixed55": kw = dict(exit_mode="fixed", tp=55, sl=145, exit_time="12:29")
            else: kw = dict(exit_mode=em)
            go(f"ORB{N} cut{cut} {em}", orb_break(g, lv, N=N, cutoff=cut, days=DISC, **kw))
for N in [1, 5, 15]:
    for k in [None, 2, 5, 10]:
        go(f"CANDLE{N} k{k}", orb_candle(g, lv, N=N, k=k, days=DISC))
df = pd.DataFrame(rows).set_index("name")
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(df[["n","win%","avg","total","maxdd","ret/dd","sharpe_ann","t","L_avg","S_avg"]].to_string())
log_variants("orb_break+candle", df.reset_index()[["name", "n", "avg", "sharpe_ann"]].to_dict("records"))
