'''
为减少时间复杂度，本任务中对数据处理部分进行拆分
数据预处理： health_dataprocess.py 选取特征，并清洗数据，生成训练数据dat1.csv和测试数据test_raw.csv
生成答案: merge.py
线下评估： evaluate.py   评估，打分
在本demo中，主要完成模型训练
其中包括特征提取，数据清洗和模型训练
仅供体验
by xrq

'''
#coding=utf-8
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('dat1.csv')   #经过预处理的全为数字的特征


#train文件为label产生文件.获取label
dat2 = pd.read_csv('meinian_round1_train_20180408.csv',encoding='gbk')
dat = pd.merge(data,dat2,on='vid')   #与训练数据融合，相当于增加了5列，分别是5个label
dat = dat.set_index('vid')
'''
手动将原文件中文特征改成了y1，y2.。即label命名为y1,y2,y3,y4,y5
'''
dat = dat.loc[(dat['y1']!='弃查') & (dat['y1']!='未查')&(dat['y2']!='弃查')&(dat['y2']!='未查')]   #清除带中文的数据
dat = dat.reset_index()
train_x = dat[['0424','2403','2404','0434','A705']].values
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
#lasso_model_1 = LinearRegression()
#lasso_model_2= LinearRegression()
#lasso_model_3 = LinearRegression()
#lasso_model_4 = LinearRegression()
#lasso_model_5 = LinearRegression()
#
#lasso_model_1.fit(train_x, label_1.ravel())
#lasso_model_2.fit(train_x, label_2.ravel())
#lasso_model_3.fit(train_x, label_3.ravel())
#lasso_model_4.fit(train_x, label_4.ravel())
#lasso_model_5.fit(train_x, label_5.ravel())


#xgboost*********************************************#
data_train_1 = xgb.DMatrix(train_x, label=label_1)

#watch_list = [(data_test_1, 'eval'), (data_train_1, 'train')]
param1 = {'max_depth': 5, 'eta': 0.6, 'silent': 1}
bst_1 = xgb.train(param1, data_train_1, num_boost_round=6)

#label2
data_train_2 = xgb.DMatrix(train_x, label=label_2)

#watch_list = [(data_test_1, 'eval'), (data_train_1, 'train')]
param2 = {'max_depth':3, 'eta': 0.6, 'silent': 1}
bst_2 = xgb.train(param2, data_train_2, num_boost_round=6)

#label3
data_train_3 = xgb.DMatrix(train_x, label=label_3)

#watch_list = [(data_test_1, 'eval'), (data_train_1, 'train')]
param3 = {'max_depth': 3, 'eta': 0.3, 'silent': 1}
bst_3 = xgb.train(param3, data_train_3, num_boost_round=6)

#label4
data_train_4 = xgb.DMatrix(train_x, label=label_4)

#watch_list = [(data_test_1, 'eval'), (data_train_1, 'train')]
param4 = {'max_depth': 2, 'eta': 0.5, 'silent': 1}
bst_4 = xgb.train(param4, data_train_4, num_boost_round=6)

#label5
data_train_5 = xgb.DMatrix(train_x, label=label_5)

#watch_list = [(data_test_1, 'eval'), (data_train_1, 'train')]
param5 = {'max_depth': 3, 'eta': 0.4, 'silent': 1}
bst_5 = xgb.train(param5, data_train_5, num_boost_round=6)

##############以下为模型预测##################
dat_test = pd.read_csv('test_raw.csv')
dat_test=dat_test[['vid','0424','2403','2404','0434','A705']]
dat_test = dat_test.set_index('vid')
#dat_test['2403'] = dat_test['2403'].fillna(dat_test['2403'].mean())   
#dat_test['424'] = dat_test['424'].fillna(dat_test['424'].mean())     
#dat_test['2404'] = dat_test['2404'].fillna(dat_test['2404'].mean())  
dat_vid = dat_test.index.values.ravel()
dat_test=dat_test.values.astype(float)
data_test = xgb.DMatrix(dat_test)
y_hat1 = bst_1.predict(data_test)
y_hat2 = bst_2.predict(data_test)
y_hat3 = bst_3.predict(data_test)
y_hat4 = bst_4.predict(data_test)
y_hat5 = bst_5.predict(data_test)


# 结果写入文件，但此时vid顺序与官方要求不一致
wf = open('ans.csv','w')
wf.write('vid,f1,f2,f3,f4,f5\n')
print 'Avalible answer: ',len(y_hat1)
for i in range(len(y_hat1)):
        wf.write('%s,%s,%s,%s,%s,%s\n'%(dat_vid[i],y_hat1[i],y_hat2[i],y_hat3[i],y_hat4[i],y_hat5[i]))
wf.close()
