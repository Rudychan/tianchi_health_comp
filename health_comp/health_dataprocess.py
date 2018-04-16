'''
产生训练数据 dat1.cav
以及测试数据 test_raw.csv
'''

import pandas as pd
import numpy as np
import re
################### 0434列处理//by 张宇 ###############
def deal_0434(s):
    s = str(s)
    list1 = [["高血压", "糖尿病", "冠心病", "脂肪肝", "肝脏脂肪"], ['血糖偏高', '血压偏高', '血脂偏高'], ['吸烟']]

    for i in range(3):
        for j in list1[i]:
            if j in s:
                return 10-5*i
    return 0


def deal_A705(s):
    
    s = str(s)
    list1 = [["肝脏脂肪"], ['肝脏硬度']]
    for i in range(2):
        for j in list1[i]:
            if j in s:
                return 10-8*i
    return 0

def deal_0424(s):
    s = str(s)
    m = re.compile(r'(\d+).*(\D)(\d+).*')
    m2 = re.compile(r'(\D)(\d+).*')
    m3 = re.compile(r'(\d+)(\D).*')
    result1 = re.search(m2,s)
    result2 = re.search(m,s)
    result3 = re.search(m3,s)
    if result1:
        return result1.group(2)
    if result2:
        return result2.group(3)
    if result3:
        return result3.group(1)
    else:
        return s
    

data = pd.read_table('meinian_round1_data_part1_20180408.txt',sep='$',)
data1 = pd.read_table('meinian_round1_data_part2_20180408.txt',sep='$')
data = data.append(data1)
data = data.dropna()  #剔除全空的数据
data = data.drop_duplicates(['vid','table_id'])   #去重，保留重复项的最后一个
data = data.reset_index()
pivoted = data.pivot('vid','table_id','field_results')  #将 table_id作为行索引进行处理 ，long-data处理/stack,unstack同理
pivoted = pivoted.dropna(axis=0, how='all')   #抛弃行全为0的
pivoted = pivoted.dropna(axis=1, how='all')    #抛弃全为0的列
dat_test = pivoted[['0424','2403','2404','0434','A705']]    #提取这三列作为特征
#dat_test.to_csv('observe.csv')
#def func1(m):
#    return m.group(1)
#m1 = re.compile(r'(\d+)--(\d+).*')
#dat_test['0424'] = dat_test['0424'].apply(lambda x:re.sub(m1,func1,str(x)))
#dat_test['0424'] = dat_test['0424'].apply(lambda x:re.sub(re.compile('\D'),'',str(x)))

#dat_test['2403'] = dat_test['2403'].apply(lambda x:re.sub(re.compile('未查'),'70',str(x)))
#dat_test['2404'] = dat_test['2404'].apply(lambda x:re.sub(re.compile('未查'),'170',str(x)))
#dat_test['2403'] = dat_test['2403'].apply(lambda x:re.sub(re.compile('None'),'70',str(x)))
#dat_test['2404'] = dat_test['2404'].apply(lambda x:re.sub(re.compile('None'),'170',str(x)))
#正则化处理，从文字种提取数据
dat_test['0434'] = dat_test['0434'].apply(deal_0434)
dat_test['A705'] = dat_test['A705'].apply(deal_A705)
dat_test['0424'] = dat_test['0424'].apply(deal_0424)

dat_test['0424'] = dat_test['0424'].apply(pd.to_numeric, errors='coerce') #强制砖换成数字，转换不成功的自动填充空白
dat_test['2403'] = dat_test['2403'].apply(pd.to_numeric, errors='coerce')
dat_test['2404'] = dat_test['2404'].apply(pd.to_numeric, errors='coerce')
dat_test['0424'] = dat_test['0424'].fillna(dat_test['0424'].mean())
dat_test['2403'] = dat_test['2403'].fillna(dat_test['2403'].mean())
dat_test['2404'] = dat_test['2404'].fillna(dat_test['2404'].mean())  #空白数据填充，本demo统一采用取均值填充
#异常值处理，对于超出范围的值直接填充均值
col_1 =dat_test['0424']
col_1[((np.abs(col_1)<50) | (np.abs(col_1)>110)) ]=np.mean(col_1)
col_2 =dat_test['2403']
col_2[((np.abs(col_2)<35) | (np.abs(col_2)>130)) ]=np.mean(col_2)
col_3 =dat_test['2404']
col_3[((np.abs(col_3)<130) | (np.abs(col_3)>195)) ]=np.mean(col_3)
dat_test.to_csv('dat1.csv',encoding='utf8')

#生成测试数据
dat2 = pd.read_csv('[new] meinian_round1_test_a_20180409.csv',encoding='gbk')
dat_test = dat_test.reset_index()
dat = pd.merge(dat_test,dat2,on='vid')   #与label文件融合人
dat_test = dat[['vid','0424','2403','2404','0434','A705']]
dat_test = dat_test.set_index('vid')
dat_test.to_csv('test_raw.csv')


#dat2 = dat1.loc[dat1['field_results'].str.isdigit()]

