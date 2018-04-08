import pandas as pd
x = pd.read_csv('tianchi_fresh_comp_train_user.csv')
x['time'] = x['time'].str.split(' ',expand = True)[0]
data = x.loc[(x['time']=='2014-12-02')|(x['time']=='2014-12-03')|(x['time']=='2014-12-04')|(x['time']=='2014-12-18')]
data.to_csv('a.csv')