import gc
import pandas as pd
import numpy as np
from scipy import sparse as ssp
import pylab as plt
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,MinMaxScaler,OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD,NMF,PCA,FactorAnalysis
from sklearn.feature_selection import SelectFromModel,SelectPercentile,f_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.cross_validation import StratifiedKFold,KFold
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint,Callback
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda,AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
from keras.models import Model
def reduce_dimen(dataset,column,toreplace):
    for index,i in dataset[column].duplicated(keep=False).iteritems():
        if i==False:
            dataset.set_value(index,column,toreplace)
    return dataset


# In[4]:

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
    #dataset = dataset.drop('date', axis = 1)

    return dataset

seed = 1
np.random.seed(seed)
dim = 32
hidden=64

path = "input/"

class AucCallback(Callback):  #inherits from Callback

    def __init__(self, validation_data=(), patience=25,is_regression=True,best_model_name='best_keras.mdl',feval='roc_auc_score',batch_size=1024*8):
        super(Callback, self).__init__()

        self.patience = patience
        self.X_val, self.y_val = validation_data  #tuple of validation X and y
        self.best = -np.inf
        self.wait = 0  #counter for patience
        self.best_model=None
        self.best_model_name = best_model_name
        self.is_regression = is_regression
        self.y_val = self.y_val#.astype(np.int)
        self.feval = feval
        self.batch_size = batch_size
    def on_epoch_end(self, epoch, logs={}):
        p = self.model.predict(self.X_val,batch_size=self.batch_size, verbose=0)#.ravel()
        if self.feval=='roc_auc_score':
            current = roc_auc_score(self.y_val,p)

        if current > self.best:
            self.best = current
            self.wait = 0
            self.model.save_weights(self.best_model_name,overwrite=True)


        else:
            if self.wait >= self.patience:
                self.model.stop_training = True
                print('Epoch %05d: early stopping' % (epoch))


            self.wait += 1 #incremental the number of times without improvement
        print('Epoch %d Auc: %f | Best Auc: %f \n' % (epoch,current,self.best))


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]



def main():

    train = pd.read_csv("input/train.csv",dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8}, parse_dates=['date'])
    test  = pd.read_csv("input/test.csv", dtype={'people_id': np.str, 'activity_id': np.str}, parse_dates=['date'])
    people    = pd.read_csv("input/people.csv", dtype={'people_id': np.str, 'activity_id': np.str, 'char_38': np.int32}, parse_dates=['date'])
    train.drop('char_10',1,inplace=True)
    test.drop('char_10',1,inplace=True)
    train = act_data_treatment(train)
    test = act_data_treatment(test)
    people = act_data_treatment(people)
    people['gf'] = 0

    people['gf'] = people.groupby(['group_1'])['gf'].transform('count')

    train['adur']=np.busday_count( train.old_date.values.astype('<M8[D]'),train.date.values.astype('<M8[D]') )

    test['adur']=np.busday_count( test.old_date.values.astype('<M8[D]'),
                                       test.date.values.astype('<M8[D]') )


    train.drop('old_date',inplace=True,axis=1)
    test.drop('old_date',inplace=True,axis=1)

    train['af']=0
    train['af'] = train.groupby(['people_id'])['af'].transform('count')
    test['af']=0
    test['af'] = test.groupby(['people_id'])['af'].transform('count')

    people['flag_count'] = people.char_11+people.char_12+people.char_13+people.char_14+people.char_15+people.char_16+people.char_17+people.char_18+people.char_19+people.char_20+people.char_21+people.char_22+people.char_23+people.char_24+people.char_25+people.char_26+people.char_27+people.char_28+people.char_29+people.char_30+people.char_31+people.char_32+people.char_33+people.char_34+people.char_35+people.char_36+people.char_37
    train = train.merge(people, on='people_id', how='left', left_index=True)
    test  = test.merge(people, on='people_id', how='left', left_index=True)

    train['bddiff']=np.busday_count(train.date_y.values.astype('<M8[D]'), train.date_x.values.astype('<M8[D]') )

    test['bddiff']=np.busday_count(test.date_y.values.astype('<M8[D]'), test.date_x.values.astype('<M8[D]') )
    gc.collect()
    train['grp_actf'] = train.groupby('group_1')['group_1'].transform('count')

    train['grp_actf_date'] = train.groupby(['group_1','date_x'])['group_1'].transform('count')

    train.drop(['date_x','date_y'],inplace=True,axis=1)

    test.drop(['date_x','date_y'],inplace=True,axis=1)
    train = train[train.grp_actf<1000]
    train_group = train.group_1.unique()
    train_group_df = pd.DataFrame({'group_1':train_group})
    train_group_df_train = train_group_df.sample(frac=0.95,random_state=127937)
    train_group_df_test = train_group_df[~train_group_df.group_1.isin(train_group_df_train.group_1.values)]

    columns = people.columns
    test['outcome'] = np.nan
    data = pd.concat([train,test])
    train = data[:train.shape[0]]
    test = data[train.shape[0]:]



    columns = train.columns.tolist()
    columns.remove('activity_id')
    columns.remove('outcome')
    data = pd.concat([train,test])
    for c in columns:
        data[c] = LabelEncoder().fit_transform(data[c].values)

    train = data[:train.shape[0]]
    test = data[train.shape[0]:]

    data = pd.concat([train,test])
    columns = train.columns.tolist()
    columns.remove('activity_id')
    columns.remove('outcome')
    flatten_layers = []
    inputs = []
    for c in columns:

        inputs_c = Input(shape=(1,), dtype='int32')

        num_c = len(np.unique(data[c].values))

        embed_c = Embedding(
                        num_c,
                        dim,
                        dropout=0.2,
                        input_length=1
                        )(inputs_c)
        flatten_c= Flatten()(embed_c)

        inputs.append(inputs_c)
        flatten_layers.append(flatten_c)

    flatten = merge(flatten_layers,mode='concat')

    fc1 = Dense(hidden,activation='relu')(flatten)
    dp1 = Dropout(0.5)(fc1)

    outputs = Dense(1,activation='sigmoid')(dp1)

    model = Model(input=inputs, output=outputs)
    model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
              )

    del data

    X = train[columns]
    X_t = test[columns].values
    X['outcome'] = train.outcome.as_matrix()
    people_id = train["people_id"].values
    activity_id = test['activity_id']
    del train
    del test
    X_train = X[X.group_1.isin(train_group_df_train.group_1.values)]
    X_test = X[~X.group_1.isin(train_group_df_train.group_1.values)]
    print X_train
    print X_test
    y_train = X_train.outcome.as_matrix()
    y_test = X_test.outcome.as_matrix()
    X_train.drop('outcome',1,inplace=True)
    X_test.drop('outcome',1,inplace=True)
    X_train = X_train.as_matrix()
    X_test = X_test.as_matrix()
    X.drop('outcome',1,inplace=True)
    X_train = [X_train[:,i] for i in range(X.shape[1])]
    X_test = [X_test[:,i] for i in range(X.shape[1])]
    del X

    model_name = 'mlp_residual_%s_%s.hdf5'%(dim,hidden)
    model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True)
    auc_callback = AucCallback(validation_data=(X_test,y_test), patience=5,is_regression=True,best_model_name=path+'best_keras.mdl',feval='roc_auc_score')

    nb_epoch = 20

    batch_size = 1024*8
    load_model = False

    if load_model:
        print('Load Model')
        model.load_weights(path+model_name)
        # model.load_weights(path+'best_keras.mdl')

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        verbose=1,
        shuffle=True,
        validation_data=[X_test,y_test],
        # callbacks = [
            # model_checkpoint,
            # auc_callback,
            # ],
        )

    # model.load_weights(model_name)
    # model.load_weights(path+'best_keras.mdl')

    y_preds = model.predict(X_test,batch_size=1024*8)
    # print('auc',roc_auc_score(y_test,y_preds))

    # print('Make submission')
    X_t = [X_t[:,i] for i in range(X_t.shape[1])]
    outcome = model.predict(X_t,batch_size=1024*8)
    submission = pd.DataFrame()
    submission['activity_id'] = activity_id
    submission['outcome'] = outcome
    submission.to_csv('submission_residual_%s_%s.csv'%(dim,hidden),index=False)

main()