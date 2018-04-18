import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # for validation 
import lightgbm as lgb
import gc # memory 
from datetime import datetime # train time checking



start = datetime.now()
VALIDATE = False
RANDOM_STATE = 50
VALID_SIZE = 0.90
MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 650
skiprows = range(1,109903891)
nrows = 75000000
output_filename = 'submission.csv'

path = '../input/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }



train_cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
train_df = pd.read_csv(path+"train.csv", skiprows=skiprows, nrows=nrows,dtype=dtypes, usecols=train_cols)

len_train = len(train_df)
gc.collect()

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]

def prep_data( df ):
    
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    
    df['in_test_hh'] = (   3 
                         - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                         - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
    gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day', 'in_test_hh'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_test_hh'})
    df = df.merge(gp, on=['ip','day','in_test_hh'], how='left')
    df.drop(['in_test_hh'], axis=1, inplace=True)
    df['nip_day_test_hh'] = df['nip_day_test_hh'].astype('uint32')
    del gp
    gc.collect()

    gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_hh'})
    df = df.merge(gp, on=['ip','day','hour'], how='left')
    df['nip_day_hh'] = df['nip_day_hh'].astype('uint16')
    del gp
    gc.collect()
    
    gp = df[['ip', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hh_os'})
    df = df.merge(gp, on=['ip','os','hour'], how='left')
    df['nip_hh_os'] = df['nip_hh_os'].astype('uint16')
    del gp
    gc.collect()

    gp = df[['ip', 'app', 'hour', 'channel']].groupby(by=['ip', 'app',  'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hh_app'})
    df = df.merge(gp, on=['ip','app','hour'], how='left')
    df['nip_hh_app'] = df['nip_hh_app'].astype('uint16')
    del gp
    gc.collect()

    gp = df[['ip', 'device', 'hour', 'channel']].groupby(by=['ip', 'device', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hh_dev'})
    df = df.merge(gp, on=['ip','device','hour'], how='left')
    df['nip_hh_dev'] = df['nip_hh_dev'].astype('uint32')
    del gp
    gc.collect()

    df.drop( ['ip','day'], axis=1, inplace=True )
    gc.collect()
    return df

train_df = prep_data(train_df)
gc.collect()

params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric':'auc',
          'learning_rate': 0.1,
          'num_leaves': 9,  # we should let it be smaller than 2^(max_depth)
          'max_depth': 5,  # -1 means no limit
          'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
          'max_bin': 100,  # Number of bucketed bin for feature values
          'subsample': 0.9,  # Subsample ratio of the training instance.
          'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
          'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
          'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
          'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
          'nthread': 8,
          'verbose': 0,
          'scale_pos_weight':99.7, # because training data is extremely unbalanced 
         }

target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'nip_day_test_hh', 'nip_day_hh', 'nip_hh_os', 'nip_hh_app', 'nip_hh_dev']
categorical = ['app', 'device', 'os', 'channel', 'hour']


if VALIDATE:

    train_df, val_df = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )
    dtrain = lgb.Dataset(train_df[predictors].values, 
                         label=train_df[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical)
    del train_df
    gc.collect()

    dvalid = lgb.Dataset(val_df[predictors].values,
                         label=val_df[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical)
    del val_df
    gc.collect()

    evals_results = {}

    model = lgb.train(params, 
                      dtrain, 
                      valid_sets=[dtrain, dvalid], 
                      valid_names=['train','valid'], 
                      evals_result=evals_results, 
                      num_boost_round=MAX_ROUNDS,
                      early_stopping_rounds=EARLY_STOP,
                      verbose_eval=50, 
                      feval=None)

    del dvalid

else:

    gc.collect()
    dtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del train_df
    gc.collect()

    evals_results = {}

    model = lgb.train(params, 
                      dtrain, 
                      valid_sets=[dtrain], 
                      valid_names=['train'], 
                      evals_result=evals_results, 
                      num_boost_round=OPT_ROUNDS,
                      verbose_eval=50,
                      feval=None)
    
del dtrain
gc.collect()

test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=test_cols)
test_df = prep_data(test_df)
gc.collect()

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']
sub['is_attributed'] = model.predict(test_df[predictors])
sub.to_csv(output_filename, index=False, float_format='%.9f')

print('=='*35)
print('============================ Final Report ============================')
print('=='*35)
print(datetime.now(), '\n')
print('{:^17} : {:}'.format('train time', datetime.now()-start))
print('{:^17} : {:}'.format('output file', output_filename))
print('{:^17} : {:.5f}'.format('train auc', model.best_score['train']['auc']))
if VALIDATE:
    print('{:^17} : {:.5f}\n'.format('valid auc', model.best_score['valid']['auc']))
    print('{:^17} : {:}\n{:^17} : {}\n{:^17} : {}'.format('VALIDATE', VALIDATE, 'VALID_SIZE', VALID_SIZE, 'RANDOM_STATE', RANDOM_STATE))
print('{:^17} : {:}\n{:^17} : {}\n{:^17} : {}\n'.format('MAX_ROUNDS', MAX_ROUNDS, 'EARLY_STOP', EARLY_STOP, 'OPT_ROUNDS', model.best_iteration))
print('{:^17} : {:}\n{:^17} : {}\n'.format('skiprows', skiprows, 'nrows', nrows))
print('{:^17} : {:}\n{:^17} : {}\n'.format('variables', predictors, 'categorical', categorical))
print('{:^17} : {:}\n'.format('model params', params))
print('=='*35)