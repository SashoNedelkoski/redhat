
import pandas as pd
import numpy as np
import datetime
from itertools import product
from scipy import interpolate ## For other interpolation functions.
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
import gc
import xgboost as xgb

def reduce_dimen(dataset,column,toreplace):
    for index,i in dataset[column].duplicated(keep=False).iteritems():
        if i==False:
            dataset.set_value(index,column,toreplace)
    return dataset

def act_data_treatment(dsname):
    dataset = dsname
    quarter ={1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4}

    for col in list(dataset.columns):
        if col not in ['people_id', 'activity_id', 'date', 'outcome','old_date']:
            if dataset[col].dtype == 'object':
                dataset[col].fillna('type 0', inplace=True)
                dataset[col] = dataset[col].apply(lambda x: x.split(' ')[1]).astype(np.int32)
            elif dataset[col].dtype == 'bool':
                dataset[col] = dataset[col].astype(np.int8)

    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset['isweekend'] = (dataset['date'].dt.weekday >= 5).astype(int)
    dataset['quarter']=dataset['month'].apply(lambda x:quarter[x])

    return dataset


act_train_data = pd.read_csv("input/train.csv",dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8}, parse_dates=['date'])
act_test_data  = pd.read_csv("input/test.csv", dtype={'people_id': np.str, 'activity_id': np.str}, parse_dates=['date'])


act_train_data=act_train_data.drop('char_10',axis=1)
act_test_data=act_test_data.drop('char_10',axis=1)

print("Train data shape: " + format(act_train_data.shape))
print("Test data shape: " + format(act_test_data.shape))

act_train_data  = act_data_treatment(act_train_data)
act_test_data   = act_data_treatment(act_test_data)

people_data    = pd.read_csv("input/people.csv", dtype={'people_id': np.str, 'activity_id': np.str, 'char_38': np.int32}, parse_dates=['date'])
people_data   = act_data_treatment(people_data)

people_data['gf'] = 0
people_data['gf'] = people_data.groupby(['group_1'])['gf'].transform('count')

act_train_data['adur']=np.busday_count( act_train_data.old_date.values.astype('<M8[D]'),
                                       act_train_data.date.values.astype('<M8[D]') )

act_test_data['adur']=np.busday_count( act_test_data.old_date.values.astype('<M8[D]'),
                                       act_test_data.date.values.astype('<M8[D]') )

act_train_data.drop('old_date',inplace=True,axis=1)
act_test_data.drop('old_date',inplace=True,axis=1)


act_train_data['af']=0
act_train_data['af'] = act_train_data.groupby(['people_id'])['af'].transform('count')
act_test_data['af']=0
act_test_data['af'] = act_test_data.groupby(['people_id'])['af'].transform('count')

people_data['flag_count'] = people_data.char_11+people_data.char_12+people_data.char_13+people_data.char_14+people_data.char_15+people_data.char_16+people_data.char_17+people_data.char_18+people_data.char_19+people_data.char_20+people_data.char_21+people_data.char_22+people_data.char_23+people_data.char_24+people_data.char_25+people_data.char_26+people_data.char_27+people_data.char_28+people_data.char_29+people_data.char_30+people_data.char_31+people_data.char_32+people_data.char_33+people_data.char_34+people_data.char_35+people_data.char_36+people_data.char_37

train = act_train_data.merge(people_data, on='people_id', how='left', left_index=True)
test  = act_test_data.merge(people_data, on='people_id', how='left', left_index=True)

train['bddiff']=np.busday_count(train.date_y.values.astype('<M8[D]'), train.date_x.values.astype('<M8[D]') )

test['bddiff']=np.busday_count(test.date_y.values.astype('<M8[D]'), test.date_x.values.astype('<M8[D]') )

train['grp_actf'] = train.groupby('group_1')['group_1'].transform('count')

train['grp_actf_date'] = train.groupby(['group_1','date_x'])['group_1'].transform('count')

train = train[train.grp_actf<1000]

gc.collect()


del act_train_data
del act_test_data
del people_data

train=train.sort_values(['people_id'],ascending=[1])
test=test.sort_values(['people_id'],ascending=[1])
y = train.outcome

train.drop(['date_x','date_y'],inplace=True,axis=1)
test.drop(['date_x','date_y'],inplace=True,axis=1)

train_group_df = pd.DataFrame({'group_1':train.group_1})


train_columns = train.columns.values
test_columns = test.columns.values
features = list(set(train_columns) & set(test_columns))
print features
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
X =train[features]
X_test=test
print 'Length of Xtest is:',len(X_test)

computed_test    = pd.read_csv("input/Submissionaid.csv")

filter_aid = computed_test.outcome.isnull()

null_aid = computed_test[filter_aid].activity_id

X_test1 = X_test[X_test.activity_id.isin(null_aid)]

print len(X_test),len(X_test1),len(null_aid)

X_test1_activity_id = X_test1.activity_id
X_test1 = X_test1[features].drop(['people_id','activity_id'], axis = 1)

from sklearn.cross_validation import LabelKFold
kf = LabelKFold(train_group_df['group_1'],n_folds=5)
i = 0
for train_index, cv_index in kf:
    X_sample_train = X.iloc[train_index]
    X_sample_eval = X.iloc[cv_index]
    X_sample_eval_act = X_sample_eval.activity_id
    del X_sample_train['activity_id']
    del X_sample_eval['activity_id']
    del X_sample_train['people_id']
    del X_sample_eval['people_id']
    print len(cv_index),len(train_index)
    dtrain = xgb.DMatrix(X_sample_train,label=y.iloc[train_index])
    dval = xgb.DMatrix(X_sample_eval,label=y.iloc[cv_index])
    param = {'max_depth':6, 'eta':0.03, 'silent':1, 'objective':'binary:logistic' }
    param['eval_metric'] = 'auc'
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.7
    param['colsample_bylevel'] = 0.7
    param['min_child_weight'] = 0
    param['booster'] = 'gbtree'
    watchlist  = [(dtrain,'train'),(dval,'eval')]
    num_round = 250
    early_stopping_rounds=10
    bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=early_stopping_rounds)
    dtest = xgb.DMatrix(X_sample_eval)
    ypred = bst.predict(dtest)
    dtest = xgb.DMatrix(X_test1)
    ypred  = bst.predict(dtest,ntree_limit=bst.best_ntree_limit)

    output = pd.DataFrame({ 'activity_id' : X_test1_activity_id, 'outcome': ypred })

    testsetdt = computed_test[~filter_aid]
    submit = pd.concat([testsetdt[['activity_id','outcome']],output[['activity_id','outcome']]], axis=0, copy=True)

    submit.to_csv('Submissiontest'+str(i)+'.csv', index=False)



    i+=1