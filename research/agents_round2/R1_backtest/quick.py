import pandas as pd, numpy as np, r1lib as R
A=pd.read_pickle('all_combos.pkl')
for (sid,name),v in A.items():
    T=v['T']; st=R.seg_table(T) if len(T) else None
    if st is None: print(name,'0'); continue
    s=" ".join(f"{k[:1]}{st[k]['n']:3d}/{st[k]['avg']:+5.0f}" if st[k]['n'] else f"{k[:1]}  0" for k in R.SEGS)
    print(f"{name:28s} {s} | 全{st['全期']['tot']:+8,.0f} dd{st['全期']['mdd']:+7,.0f} skips{v['skips']}")
