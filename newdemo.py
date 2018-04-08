#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

def load_data(path):
    #取出数据 ，并把时间变成标准格式
    data = pd.read_csv(path,index_col=['user_id','item_id'])     # totally 1000 row data
    action = pd.get_dummies(data.behavior_type)    #one-hot coding
    action = action.rename(columns=lambda x: 'action' + str(x))  # rename the column by number
    data = pd.concat([data,action],axis = 1)   
    x =data[['item_category','time','action1','action2','action3','action4']]
#    x['time'] = x['time'].str.split(' ',expand = True)[0]   
    return x
    
def extract_data(x,time,flag):    
#     从目标日期抽取出目标样本的uer-item对 ,将其他值加和.返回值是uid对和一个聚合的元祖 
    if flag==0:
        x_train = x.loc[(x['time']==time)]
    if flag==1:
        x_train = x.loc[(x['time']==time) |(x['time']=='2014-12-02')]
    x_train = x_train.groupby(['user_id','item_id'])[['action1','action2','action3','action4']].agg('sum')
    pair = x_train.index.values
    label_pair = []
    for i in pair:
        label_pair.append(i)
    return label_pair,x_train

def label(time):
# 完成label提取，从下一日发pair减去无关的pair，然后提取action4作为上一日的label
    pair,dat = extract_data(data,time,0)
#    print dat.loc[(dat['action4']==1)]     

    y_pair = []
    a=set(pair)
    b=set(uid)
    d = list(a&b)

    for i in d:   
        if dat.xs(i)['action4']==1 :
            y_pair.append(i)
#        y_train.append(int(dat.xs(i)['action4']))
    return y_pair
#    print len(dat)
        
# 将并集在原样本中进行检索，替换action4的选项，其余选项设为00000  
    
def train_data() :
    x['action4']=0
#    data_del=[]
    dat1 = x.loc[(x['action1']>2) | (x['action2']!=0) | (x['action3']!=0)]

    dat = dat1.reset_index()
    dat = dat.values
    num= []
    for line in dat:
        for j in label_pair:
            if line[0]==j[0] and line[1]==j[1]:
                line[5]=1
                num.append('*')       #样本中购买的数量
    print 'Purchase in sample:',len(num)

    np.set_printoptions(threshold=1000)
    label_y =  dat[:,5]
    train_x = dat[:,2:5]
   
    return label_y,train_x
#    print label_pair

#    print extract
#    
#    for line in label_pair:
#        if (dat['user_id']==line[0] and dat['item_id']==line[1]):
#            dat['action4']=1
#    print dat
#                   
##    x=x.loc[(x['time']=='2014-12-02') | (x['time']=='2014-12-01') | (x['time']=='2014-12-10')]
##    label = x[['action4']]
def train():
    model = Pipeline([
        ('ss', StandardScaler()),
        ('clf', LogisticRegression())])
    model = model.fit(x_train, y_train)
#    reg = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1)
#    dt = reg.fit(x_train, y_train)
#    y_hat = dt.predict(x_test)
#    print y_train

#    y_hat = model.predict(x_test)
    py = model.predict_proba(x_test)
    npy  =[]
    for a in py:
        npy.append(a[1])
    py = npy

    
    
    pro_pair=val.index.values
    lx = zip(pro_pair,py)
    lx = sorted(lx,key=lambda x:x[1],reverse = True)
    y_hat_uid=[]
    predict=[]
    item_set = pd.read_csv('tianchi_fresh_comp_train_item.csv')
    item_set1 = item_set['item_id'].values
    answer_num =100
    print 'Answer number: ',answer_num
    for i in range(answer_num):
        item = lx[i];
        predict.append(item[0][1])
    a3 = set(predict)
    b3 = set(item_set1)
    c3 = list(a3 & b3)
    for j in range(answer_num):
        item2 = lx[j]
        for k in c3:
            print 'K:',k,item2
            if k==item2[0][1]:
                y_hat_uid.append((item2[0][0],item2[0][1]))
                print '******************************',(item2[0][0],item2[0][1])
                break
    print 'y_hat_uid',y_hat_uid
#        for j in range(len(item_set1)):
#            if item[0][1]==item_set1[j]:
#                y_hat_uid.append((item[0][0],item[0][1]))

# 从第1天的datafame提取预测值的uid
#    val['action4']=y_hat
#    dat = val.loc[(val['action4']==1)]

#    pair = dat.index.values
#    for j in pair:
#        y_hat_uid.append(j)
#    print 'y_hat_uid: \n',y_hat_uid,len(y_hat_uid)
#    print 'result_uid:\n',result_uid,len(result_uid)
    
    set1 = set(y_hat_uid)
    set2 = set(result_uid)
    d = list(set1&set2)
    print 'hit_number: \n',len(d)
    print 'Result_uid_length:',len(result_uid)
    print 'Test_set_length',len(x_test)
    return y_hat_uid,d,answer_num

  
    
def test_set(time1,time2):
    uid,x2 = extract_data(data,time1,0)
    uid2,x3 = extract_data(data,time2,0)
    x3_p = x3.loc[(x3['action4']==1)]
    t_uid=[]
    pair = x3_p.index.values
    for j in pair:
        t_uid.append(j)
    dat2 = x2.loc[(x2['action1']>3) | (x2['action2']!=0) | (x2['action3']!=0)]
    x_test = dat2.values[:,0:3]
    return x_test,uid,t_uid,dat2

def F1_val():
   if len(hit) > 0:
	a=len(result_uid)
	b=len(answer)
	c=len(hit)
	R=1.0*c/a*100
	P=1.0*c/b*100
	F1=2.0*R*P/(R+P)
	print 'F1/P/R %.2f%%/%.2f%%/%.2f%%\n' %(F1,P,R)
    
def save_answer():

#    print item_set1
#    a3= set(item_set1)
#    b3 = set(answer[:,0])
#    c3 = a3 &b3
    wf = open('ans.csv','w')
    wf.write('user_id, item_id\n')
    print 'Avalible answer: ',len(answer)
    for i in answer:
        item = i
        wf.write('%s,%s\n'%(item[0],item[1]))
    
    
if __name__=='__main__':
    data = load_data('a.csv')
    uid,x = extract_data(data,'2014-12-03',1)  # day
    #加载数据，返回当天的uid对和一个mutiindex的datafame，index是user_id 和item_id
    label_pair = label('2014-12-04')   # day+1
    #得到下一日的uid，与前一天做交集，返回action4=1的uid对
    y_train,x_train = train_data()
    #将上一步中的目标uid的action4拷贝进原数据的action4一列 ，构成完整的训练数据
    #返回label和特征
    x_test,test_uid,result_uid,val = test_set('2014-12-18','2014-12-19')  # day+1
    #生成测试集的特征部分。返回值是测试集特征、测试集所有uid、测试集结果uid，第1天的datafame

    answer,hit,answer_num=train()
    #训练和预测，返回值： 预测的uid对，预测正确的uid对，预测数量
    F1_val()
    save_answer()
#    print uid
#    print x
    
    
