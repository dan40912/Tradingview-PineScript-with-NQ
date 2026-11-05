"""04: A2_TREND_EXT_FOLLOW (frozen by A2) broken down by value-area context. Discovery only.
NOTE: A2 already reported its validation incl. on/off-I2-day split -> any filter here is contaminated
w.r.t. on/off-I2 validation; other VA splits were not reported by A2."""
import sys, os, numpy as np, pandas as pd
import c2lib as L
from lib import sim
sys.path.insert(0, os.path.join(L.RESEARCH, "agents", "A2"))
import importlib.util
spec = importlib.util.spec_from_file_location("a2c", os.path.join(L.RESEARCH, "agents", "A2", "candidates.py"))
a2c = importlib.util.module_from_spec(spec); spec.loader.exec_module(a2c)
g, lv, i2 = L.load()
fam = "04_trendext_ctx"
tr = sim.run_signals(g, a2c.trend_ext_follow(g, lv))
lvi = lv.set_index("i")
tr["i2dir"] = tr.i.map(lvi.i2_dir)
tr["typ"] = np.where(tr.i2dir == 0, "inside", np.where(tr.date.isin(set(i2.date)), "out_I2", "out_noI2"))
tr["rel"] = np.where(tr.i2dir == 0, "inside", np.where(tr.dir == tr.i2dir, "toward", "away"))
c = tr.entry  # entry price vs VA
tr["loc"] = np.where(c > tr.i.map(lvi.vah), "aboveVAH", np.where(c < tr.i.map(lvi.val), "belowVAL", "inVA"))
tr["dloc"] = tr["loc"] + np.where(tr.dir == 1, "_long", "_short")
L.disc(g, tr, "A2 trend_ext (frozen)", fam)
print("corr disc", L.corr_i2(g, tr, i2, "discovery"))
for col in ["typ", "rel", "dloc"]:
    for k, x in tr.groupby(col):
        L.disc(g, x, f"  {col}={k}", fam)
