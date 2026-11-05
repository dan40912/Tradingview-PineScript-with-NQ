import warnings; warnings.filterwarnings("ignore")
from a1lib import *
from signals import *
g, lv = load(); d = daytable(g, lv); DISC = (g.split == "discovery")
rows = []
for level in ["pd", "on"]:
    for mode in ["go", "fade"]:
        for st in [0.15, 0.3]:
            for xt in ["12:29", "15:59"]:
                tr = sim.run_signals(g, level_break(g, lv, d, level, mode, stop_atr=st, exit_time=xt, days=DISC))
                s = summ(tr, g, "discovery"); s["name"] = f"{level} {mode} st{st} exit{xt}"
                if len(tr):
                    h = halves(tr); s["h"] = "/".join(str(int(x)) for x in h["mean"].values)
                rows.append(s)
df = pd.DataFrame(rows).set_index("name")
pd.set_option("display.width", 250)
print(df[["n","avg","total","maxdd","ret/dd","sharpe_ann","t","h"]].to_string())
log_variants("pd_on_levels", df.reset_index()[["name","n","avg","sharpe_ann"]].to_dict("records"))
