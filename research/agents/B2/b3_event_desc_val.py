"""B3: descriptive check of event-day morning range (09:30-12:29) / ATR20 in discovery vs validation (no strategy selection)."""
import warnings; warnings.filterwarnings("ignore")
from util import *
g, lv, rf, ev = load()
rng = np.nanmax(g.H[:, 930:1110], 1) - np.nanmin(g.L[:, 930:1110], 1)
x = rng / rf.atr20.values
rows = []
for sp in ["discovery", "validation"]:
    base = g.tradable & (g.split == sp)
    for e in ["fomc", "fomc_next", "cpi", "nfp", "rel830", "opex", "month_start", "post_holiday"]:
        m = base & ev[e].values
        rows.append({"split": sp, "event": e, "days": int(m.sum()), "rng/atr_event": round(np.nanmedian(x[m]), 2),
                     "rng/atr_other": round(np.nanmedian(x[base & ~ev[e].values]), 2)})
print(pd.DataFrame(rows).to_string())
