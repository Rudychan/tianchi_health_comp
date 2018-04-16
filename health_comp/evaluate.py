'''
在本demo中，是正式demo的copy
用于评分以及测试
by xrq

'''
#coding=utf-8
import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
###############0434既往病史特征处理############


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

#dat.to_csv('train_dat.csv')
train_x = dat[['0424','2403','2404','0434','A705']].values
label_1 = dat[['y1']].values.astype(float)
label_2 = dat[['y2']].values.astype(float)
label_3 = dat[['y3']].values.astype(float)
label_4 = dat[['y4']].values.astype(float)
label_5 = dat[['y5']].values.astype(float)  #数据类型转换/若转换不成功则可以发现异常格式的数据
#print train_x
#
#print len(label_1)
#划分训练集和测试集
x1_train, x1_test, y1_train, y1_test = train_test_split(train_x, label_1.ravel(),test_size=0.3,random_state=1)
x2_train, x2_test, y2_train, y2_test = train_test_split(train_x, label_2.ravel(),test_size=0.3,random_state=1)
x3_train, x3_test, y3_train, y3_test = train_test_split(train_x, label_3.ravel(),test_size=0.3,random_state=1)
x4_train, x4_test, y4_train, y4_test = train_test_split(train_x, label_4.ravel(),test_size=0.3,random_state=1)
x5_train, x5_test, y5_train, y5_test = train_test_split(train_x, label_5.ravel(),test_size=0.3,random_state=1)

#xgboost*********************************************#
data_train_1 = xgb.DMatrix(x1_train, label=y1_train)
data_test_1 = xgb.DMatrix(x1_test, label=y1_test)
#watch_list = [(data_test_1, 'eval'), (data_train_1, 'train')]
param1 = {'max_depth': 5, 'eta': 0.6, 'silent': 1}
bst_1 = xgb.train(param1, data_train_1, num_boost_round=6)

#label2
data_train_2 = xgb.DMatrix(x2_train, label=y2_train)
data_test_2 = xgb.DMatrix(x2_test, label=y2_test)
#watch_list = [(data_test_1, 'eval'), (data_train_1, 'train')]
param2 = {'max_depth':3, 'eta': 0.6, 'silent': 1}
bst_2 = xgb.train(param2, data_train_2, num_boost_round=6)

#label3
data_train_3 = xgb.DMatrix(x3_train, label=y3_train)
data_test_3 = xgb.DMatrix(x3_test, label=y3_test)
#watch_list = [(data_test_1, 'eval'), (data_train_1, 'train')]
param3 = {'max_depth': 3, 'eta': 0.3, 'silent': 1}
bst_3 = xgb.train(param3, data_train_3, num_boost_round=6)

#label4
data_train_4 = xgb.DMatrix(x4_train, label=y4_train)
data_test_4 = xgb.DMatrix(x4_test, label=y4_test)
#watch_list = [(data_test_1, 'eval'), (data_train_1, 'train')]
param4 = {'max_depth': 2, 'eta': 0.5, 'silent': 1}
bst_4 = xgb.train(param4, data_train_4, num_boost_round=6)

#label5
data_train_5 = xgb.DMatrix(x5_train, label=y5_train)
data_test_5 = xgb.DMatrix(x5_test, label=y5_test)
#watch_list = [(data_test_1, 'eval'), (data_train_1, 'train')]
param5 = {'max_depth': 3, 'eta': 0.4, 'silent': 1}
bst_5 = xgb.train(param5, data_train_5, num_boost_round=6)


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
#lasso_model_1.fit(x1_train, y1_train)
#lasso_model_2.fit(x2_train, y2_train)
#lasso_model_3.fit(x3_train, y3_train)
#lasso_model_4.fit(x4_train, y4_train)
#lasso_model_5.fit(x5_train, y5_train)


##############以下为模型预测##################
dat_test = pd.read_csv('test_raw.csv')
dat_test=dat_test[['vid','0424','2403','2404','0434','A705']]
dat_test = dat_test.set_index('vid')
dat_test['2403'] = dat_test['2403'].fillna(dat_test['2403'].mean())   
dat_test['0424'] = dat_test['0424'].fillna(dat_test['0424'].mean())     
dat_test['2404'] = dat_test['2404'].fillna(dat_test['2404'].mean())  
dat_vid = dat_test.index.values.ravel()
dat_test=dat_test.values.astype(float)

#y_hat1 = lasso_model_1.predict(x1_test)
#y_hat2 = lasso_model_2.predict(x2_test)
#y_hat3 = lasso_model_3.predict(x3_test)
#y_hat4 = lasso_model_4.predict(x4_test)
#y_hat5 = lasso_model_5.predict(x5_test)

y_hat1 = bst_1.predict(data_test_1)
y_hat2 = bst_2.predict(data_test_2)
y_hat3 = bst_3.predict(data_test_3)
y_hat4 = bst_4.predict(data_test_4)
y_hat5 = bst_5.predict(data_test_5)
print y_hat3
print y_hat2
#以下用于打分

e1 = np.square(np.log((y_hat1+1) / (y1_test+1)))
e1 = np.mean(e1)
print 'e1: ',e1
e2 = np.square(np.log((y_hat2+1) / (y2_test+1)))
e2 = np.mean(e2)
print 'e2: ',e2
e3 = np.square(np.log((y_hat3+1) / (y3_test+1)))
e3 = np.mean(e3)
print 'e3: ',e3
e4 = np.square(np.log((y_hat4+1) / (y4_test+1)))
e4 = np.mean(e4)
print 'e4: ',e4
e5 = np.square(np.log((y_hat5+1) / (y5_test+1)))
e5 = np.mean(e5)
print 'e5: ',e5
e = (e1+e2+e3+e4+e5) / 5
print 'Score: ',e






# 结果写入文件，但此时vid顺序与官方要求不一致
#wf = open('ans.csv','w')
#wf.write('vid,f1,f2,f3,f4,f5\n')
#print 'Avalible answer: ',len(y_hat1)
#for i in range(len(y_hat1)):
#        wf.write('%s,%s,%s,%s,%s\n'%(y_hat1[i],y_hat2[i],y_hat3[i],y_hat4[i],y_hat5[i]))
#wf.close()
