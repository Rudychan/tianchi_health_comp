import pandas as pd
import re
data = pd.read_table('meinian_round1_data_part1_20180408.txt',sep='$',index_col=['vid'])
data1 = pd.read_table('meinian_round1_data_part2_20180408.txt',sep='$',index_col=['vid'])
data = data.append(data1)

#以上为原始数据
data= data.reset_index()
data = data.drop_duplicates(['vid','table_id'])   #去重
data = data.pivot('vid','table_id','field_results')  #将 table_id作为列进行处理
dat1= data.reset_index()
dat2 = pd.read_csv('[new] meinian_round1_test_a_20180409.csv',encoding='gbk')
dat = pd.merge(dat1,dat2,on='vid')   #与label文件融合人
dat_test = dat[['vid','0424','2403','2404']]
dat_test = dat_test.set_index('vid')
def func1(m):
    return m.group(1)
m1 = re.compile(r'(\d+)--(\d+).*')

dat_test['0424'] = dat_test['0424'].apply(lambda x:re.sub(m1,func1,str(x)))
dat_test['0424'] = dat_test['0424'].apply(lambda x:re.sub(re.compile('\D'),'',str(x)))
dat_test['2403'] = dat_test['2403'].apply(lambda x:re.sub(re.compile('未查'),'70',str(x)))
dat_test['2404'] = dat_test['2404'].apply(lambda x:re.sub(re.compile('未查'),'170',str(x)))
dat_test['2403'] = dat_test['2403'].apply(lambda x:re.sub(re.compile('None'),'70',str(x)))
dat_test['2404'] = dat_test['2404'].apply(lambda x:re.sub(re.compile('None'),'170',str(x)))

#print dat_test
dat_test.to_csv('test_raw.csv')

#dat = dat.pivot('vid','table_id','field_results')  #将 table_id作为列进行处理
