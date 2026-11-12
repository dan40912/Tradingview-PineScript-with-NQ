"""檢查 2026-11-11 晚上（交易日 11/12 的亞洲盤，資料到 19:55）有沒有亞洲做多訊號。研究用途，不是投資建議。"""
import warnings; warnings.filterwarnings("ignore")
from core import *
from lib.features import value_area
from lib import sim
ge = data.load_grid("ES", include_holdout=True)
def asia_long(G, date_prev, date):
    i, j = G.dindex[date_prev], G.dindex[date]
    rH, rL, rV = G.rth(G.H), G.rth(G.L), G.rth(G.V)
    val, poc, vah = value_area(rH[i], rL[i], rV[i]); pdl = np.nanmin(rL[i])
    out = {}
    for name, lvl in [("ASIA-PD-k1 多", pdl), ("ASIA-VA-k1 多", val)]:
        swept = False
        for a in range(0, sim.slot_of("02:00") + 1, 5):
            c = G.C[j][a:a + 5]; ok = ~np.isnan(c)
            if not ok.any(): continue
            if np.nanmin(G.L[j][a:a + 5]) < lvl: swept = True
            if swept and c[ok][-1] > lvl and 5 <= a + 5 <= sim.slot_of("02:00"):
                out[name] = (a + 5, sim.hhmm_of(a + 5), lvl); break
        out.setdefault(name, (None, None, lvl))
    return out
print("NQ", asia_long(g, "2026-11-11", "2026-11-12"))
print("ES", asia_long(ge, "2026-11-11", "2026-11-12"))
