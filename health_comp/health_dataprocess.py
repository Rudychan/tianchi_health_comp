import pandas as pd
import numpy as np
data = pd.read_table('meinian_round1_data_part1_20180408.txt',sep='$',index_col=['vid'])
data1 = pd.read_table('meinian_round1_data_part2_20180408.txt',sep='$',index_col=['vid'])
data = data.append(data1)
dat1 = data.sort_index()
dat1 = dat1.dropna()
dat2 = dat1.loc[dat1['field_results'].str.isdigit()]
dat2.to_csv('dat1.csv',encoding='utf8')
