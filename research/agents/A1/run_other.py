import warnings; warnings.filterwarnings("ignore")
from a1lib import *
from signals import *
g, lv = load(); d = daytable(g, lv); DISC = (g.split == "discovery")
rows = []
def go(fam, name, sigs):
    tr = sim.run_signals(g, sigs)
    s = summ(tr, g, "discovery"); s["name"] = name; s["fam"] = fam
    if len(tr):
        h = halves(tr); s["h"] = "/".join(str(int(x)) for x in h["mean"].values)
        s["L"] = round(tr[tr.dir==1].pnl.mean(),0); s["S"] = round(tr[tr.dir==-1].pnl.mean(),0)
    rows.append(s)
sig = noise_bands(g)
for mult in [0.75, 1.0, 1.5]:
    for exin in [True, False]:
        for first in ["10:00", "11:00"]:
            go("noise", f"noise m{mult} exin{exin} first{first}", noise_breakout(g, lv, sig, mult=mult, exit_inside=exin, first=first, days=DISC))
for at in ["11:00", "12:00", "13:00", "14:00"]:
    for thr in [0.3, 0.5, 0.8]:
        for st in [0.2, 0.4]:
            go("trend_pm", f"trendPM {at} thr{thr} st{st}", trend_afternoon(g, lv, d, at=at, thr=thr, stop_atr=st, days=DISC))
df = pd.DataFrame(rows).set_index("name")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(df[["n","avg","total","maxdd","ret/dd","sharpe_ann","t","L","S","h"]].to_string())
log_variants("noise+trendPM", df.reset_index()[["name","n","avg","sharpe_ann"]].to_dict("records"))
