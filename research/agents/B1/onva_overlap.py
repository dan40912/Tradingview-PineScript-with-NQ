from common import *
import pickle
g, lv = load("NQ")
res = pickle.load(open(os.path.join(HERE, "_cand_trades.pkl"), "rb"))
i2, on = res["I2_v1"], res["B1_ONVA_I2"]
m = on.merge(i2, on="i", suffixes=("_on", "_i2"))
for sp in ("discovery", "validation"):
    t = m[m.split_on == sp]
    print(sp, "days", len(t), "ONVA entry avg", round(t.pnl_on.mean()), "I2 entry avg same days", round(t.pnl_i2.mean()),
          "ONVA earlier %", round((t.entry_slot_on < t.entry_slot_i2).mean()*100), "win on", round((t.pnl_on>0).mean()*100))
    # I2 days not in ONVA
    rest = i2[(i2.split == sp) & ~i2.i.isin(on.i)]
    print("   I2 days without ONVA trade:", len(rest), round(rest.pnl.mean()))
