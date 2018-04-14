# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:21:15 2018

@author: a7419
"""

import pandas as pd
dat1 = pd.read_csv('ans.csv')
dat2 = pd.read_csv('[new] meinian_round1_test_a_20180409.csv',encoding='gbk')
dat = pd.merge(dat2,dat1)   #与label文件融合人
dat = dat[['vid','f1','f2','f3','f4','f5']]
dat= dat.set_index('vid')
dat_vid = dat.index.values.ravel()
y_hat1 = dat['f1'].values.ravel().astype(float)
y_hat2 = dat['f2'].values.ravel().astype(float)
y_hat3 = dat['f3'].values.ravel().astype(float)
y_hat4 = dat['f4'].values.ravel().astype(float)
y_hat5 = dat['f5'].values.ravel().astype(float)

wf = open('result.csv','w')
print 'Avalible answer: ',len(y_hat1)
for i in range(len(y_hat1)):
        wf.write('%s,%.3f,%.3f,%.3f,%.3f,%.3f\n'%(dat_vid[i],y_hat1[i],y_hat2[i],y_hat3[i],y_hat4[i],y_hat5[i]))
wf.close()


