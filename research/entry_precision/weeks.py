import pandas as pd, numpy as np
exec(open('book6.py').read().split("if __name__")[0].replace("NQ, ES = sys.argv[1], sys.argv[2]","NQ, ES='nq17.csv','es17.csv'"))
A=build(n); C=build(n,'高勝率')
B=n[(n.tier==1)&(n.inPlan==1)]; D=n[n.tier==2]; E=pd.concat([A,D])
books={'A 第一層六點（TradingView 版，含 ES）':A,'B Tradovate 彩色點（不含 ES）':B,'C 高勝率（只做多）':C,'D 第二層紙上 C1/TD10（5 MNQ）':D,'E 第一層六點＋第二層':E}
W=[('上週 11/09～11/13','2026-11-09','2026-11-13'),('本週 11/16～11/17(進行中)','2026-11-16','2026-11-17'),('10/01～11/17 全部','2026-10-01','2026-11-17')]
rows=[]
for bn,b in books.items():
    for wn,a,z in W:
        x=b[(b.date>=a)&(b.date<=z)]; done=x[x.xt.notna()]
        rows.append(dict(組合=bn,期間=wn,筆數=len(x),已平倉=len(done),勝=int((done.pnl>0).sum()),敗=int((done.pnl<=0).sum()),勝率=f'{(done.pnl>0).mean():.0%}' if len(done) else '-',合計=round(done.pnl.sum()),持倉中=len(x)-len(done)))
pd.set_option('display.width',250); print(pd.DataFrame(rows).to_string(index=False))
# detail
allk=pd.concat([n[(n.tier==1)|(n.tier==2)|(n.kind.isin(ESK))]])
x=allk[(allk.date>='2026-11-09')].copy()
x['inA']=x.set_index(['date','kind','et']).index.isin(A.set_index(['date','kind','et']).index)
x['入場ET']=x.et.dt.tz_convert('America/New_York').dt.strftime('%m/%d %H:%M'); x['出場ET']=x.xt.dt.tz_convert('America/New_York').dt.strftime('%m/%d %H:%M')
x['持有分']=((x.xt-x.et).dt.total_seconds()/60).round()
x['層']=np.where(x.tier==2,'第二層',np.where(x.inA,'第一層計畫內','不做'))
x['不做原因']=np.where(x.inA|(x.tier==2),'',np.where(x.kind.isin(ESK)&~x.esok,'ES沒同步',x.skipWhy.fillna('')))
print(x[['date','層','kind','dir','入場ET','entry','出場ET','exit','why','pnl','mae','mfe','持有分','不做原因']].to_string(index=False))
x.to_csv('weeks_detail.csv',index=False)
