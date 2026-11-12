import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'../B_exits')
import core as Bc, numpy as np, pandas as pd
R=Bc.simulate(Bc.plan()); Bk=Bc.book(R)
st=Bc.stats(Bk); print({k:(v['n'],round(v['tot']),round(v['mdd'])) for k,v in st.items()})
print(R.columns.tolist()); print(R.head())
print(Bc.CAND[['date','key','entry_slot','exit_slot','dir']].head())
print(Bc.CAND.exit_slot.value_counts())
