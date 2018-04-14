'''
为减少时间复杂度，本任务中对数据处理部分进行拆分
数据预处理： health_dataprocess.py ，将所有数字类型的特征保留，生成dat1.csv。
测试集生成： gen_test_set.py , 对测试数据进行清洗，生成二维测试集矩阵，生成test_raw.csv。
生成答案: merge.py

在本demo中，主要完成模型训练
其中包括特征提取，数据清洗和模型训练
仅供体验
by xrq

'''
#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('dat1.csv')   #经过预处理的全为数字的特征
data = data.dropna()  #剔除全空的数据
data = data.drop_duplicates(['vid','table_id'],keep='last')   #去重，保留重复项的最后一个
pivoted = data.pivot('vid','table_id','field_results')  #将 table_id作为行索引进行处理 ，long-data处理/stack,unstack同理
pivoted = pivoted.dropna(axis=0, how='all')   #抛弃行全为0的
pivoted = pivoted.dropna(axis=1, how='all')    #抛弃全为0的列
pivoted = pivoted[['0424','2403','2404']]   #提取这三列作为特征
pivoted['0424'] = pivoted['0424'].fillna(pivoted['0424'].mean())
pivoted['2403'] = pivoted['2403'].fillna(pivoted['2403'].mean())
pivoted['2404'] = pivoted['2404'].fillna(pivoted['2404'].mean())  #空白数据填充，本demo统一采用取均值填充
pivoted=pivoted.reset_index() 

#train文件为label产生文件.获取label
dat2 = pd.read_csv('meinian_round1_train_20180408.csv',encoding='gbk')
dat = pd.merge(pivoted,dat2,on='vid')   #与训练数据融合，相当于增加了5列，分别是5个label
dat = dat.set_index('vid')
'''
手动将原文件中文特征改成了y1，y2.。即label命名为y1,y2,y3,y4,y5
'''
dat = dat.loc[(dat['y1']!='弃查') & (dat['y1']!='未查')&(dat['y2']!='弃查')&(dat['y2']!='未查')]   #清除带中文的数据
dat = dat.reset_index()
train_x = dat[['0424','2403','2404']].values
label_1 = dat[['y1']].values.astype(float)
label_2 = dat[['y2']].values.astype(float)
label_3 = dat[['y3']].values.astype(float)
label_4 = dat[['y4']].values.astype(float)
label_5 = dat[['y5']].values.astype(float)  #数据类型转换/若转换不成功则可以发现异常格式的数据

#以下是模型训练
#model = Ridge()    # 此处选用ridge模型
#alpha_can = np.logspace(-3, 2, 10)  #交叉验证参数产生
#lasso_model_1 = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
#lasso_model_2 = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
#lasso_model_3 = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
#lasso_model_4 = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
#lasso_model_5 = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
lasso_model_1 = LinearRegression()
lasso_model_2= LinearRegression()
lasso_model_3 = LinearRegression()
lasso_model_4 = LinearRegression()
lasso_model_5 = LinearRegression()

lasso_model_1.fit(train_x, label_1.ravel())
lasso_model_2.fit(train_x, label_2.ravel())
lasso_model_3.fit(train_x, label_3.ravel())
lasso_model_4.fit(train_x, label_4.ravel())
lasso_model_5.fit(train_x, label_5.ravel())


##############以下为模型预测##################
dat_test = pd.read_csv('test_raw.csv')
dat_test=dat_test[['vid','424','2403','2404']]
dat_test = dat_test.set_index('vid')
dat_test['2403'] = dat_test['2403'].fillna(dat_test['2403'].mean())   
dat_test['424'] = dat_test['424'].fillna(dat_test['424'].mean())     
dat_test['2404'] = dat_test['2404'].fillna(dat_test['2404'].mean())  
dat_vid = dat_test.index.values.ravel()
dat_test=dat_test.values.astype(float)

y_hat1 = lasso_model_1.predict(dat_test)
y_hat2 = lasso_model_2.predict(dat_test)
y_hat3 = lasso_model_3.predict(dat_test)
y_hat4 = lasso_model_4.predict(dat_test)
y_hat5 = lasso_model_5.predict(dat_test)


# 结果写入文件，但此时vid顺序与官方要求不一致
wf = open('ans.csv','w')
wf.write('vid,f1,f2,f3,f4,f5\n')
print 'Avalible answer: ',len(y_hat1)
for i in range(len(y_hat1)):
        wf.write('%s,%s,%s,%s,%s,%s\n'%(dat_vid[i],y_hat1[i],y_hat2[i],y_hat3[i],y_hat4[i],y_hat5[i]))
wf.close()
