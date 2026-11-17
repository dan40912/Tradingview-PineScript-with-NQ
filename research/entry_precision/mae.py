import pandas as pd, numpy as np
exec(open('book6.py').read().split("if __name__")[0].replace("NQ, ES = sys.argv[1], sys.argv[2]","NQ, ES='nq17.csv','es17.csv'"))
A=build(n); A=A[A.xt.notna()].copy()
A['P']=np.select([A.date<='2025-06-30',A.date<='2025-12-31',A.date<='2026-09-30'],['1探索','2驗證','3測試'],'4近期10/1起')
A['win']=A.pnl>0; A['side']=np.where(A.dir==1,'多','空')
A['mins']=(A.xt-A.et).dt.total_seconds()/60
def q(x): return pd.Series(dict(筆數=len(x),勝率=f'{x.win.mean():.0%}',
  贏單中位逆向=round(x[x.win].mae.median()),贏單逆向超50點=f'{(x[x.win].mae>50).mean():.0%}',贏單逆向超100點=f'{(x[x.win].mae>100).mean():.0%}',
  輸單中位最多順向=round(x[~x.win].mfe.median()) if (~x.win).any() else None,輸單曾到正30=f'{(x[~x.win].mfe>=30).mean():.0%}' if (~x.win).any() else '-',
  贏單中位持有分=round(x[x.win].mins.median())))
pd.set_option('display.width',250)
print(A.groupby('P').apply(q).to_string()); print(A.groupby('side').apply(q).to_string()); print(A.groupby('kind').apply(q).to_string())
print('全部', q(A).to_dict())
A.to_csv('A_book_mae.csv',index=False)
