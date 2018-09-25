

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgbm
import Data_preprocessing as dp
import gc
from sklearn.model_selection import train_test_split

### function that change data type to prevent memory surge from Lightgbm python package
def change_data_type(df):
  df['passenger_count'] = df['passenger_count'].values.astype(np.float32)
  df['year'] = df['year'].values.astype(np.float32)
  df['month'] = df['month'].values.astype(np.float32)
  df['day'] = df['day'].values.astype(np.float32)
  df['hour'] = df['hour'].values.astype(np.float32)
  df['dayOfWeek'] = df['dayOfWeek'].values.astype(np.float32)
  df['pickup_area'] = df['pickup_area'].values.astype(np.float32)
  df['dropoff_area'] = df['dropoff_area'].values.astype(np.float32)

  return df

def train()
  
  ### load training data
  data =  pd.read_csv('cleaned_train.csv')
  
  ### preprocess data
  X, y = dp.preprocess_data(data)
  
  ### change data type
  X = change_data_type(X)
  
  ### split training set and validation set
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
  
  ### clear up memory
  del X
  del y
  del data
  gc.collect()
  
  ### parameter for lightgbm model
  params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread':16,
        'num_leaves': 31,
        'learning_rate': 0.03,
        'max_depth': 500,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 30,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': False,
        'seed':0,
        'num_rounds'60000,
    }
  
  ### create lightgbm dataset
  train_set = lgbm.Dataset(X_train, y_train, silent=False, categorical_feature=['year','month','day','dayOfWeek','pickup_area','dropoff_area'])
  test_set = lgbm.Dataset(X_test, y_test, silent=False, categorical_feature=['year','month','day','dayOfWeek','pickup_area','dropoff_area'])
  
  ### train the model
  model = lgbm.train(params, train_set = train_set, valid_sets=test_set, num_boost_round=10000,early_stopping_rounds=500,verbose_eval=100)
  
  ### save trained model
  with open("lightbgm_NYC.pkl", 'wb') as pickle_file:
    pickle.dump(model, pickle_file)
    
def test():
  test_df =  pd.read_csv('test.csv')
  key = test_df['key']
  X_test = preprocess_data(test_df)
  X_test = dp.preprocess_data(X_test)
  X_test = change_data_type(X_test)

  ### load model
  with open("lightbgm_NYC.pkl", 'rb') as pickle_file:
    model = pickle.load(pickle_file)
    
  ### predict value
  prediction = model.predict(X_kag_test, num_iteration = model.best_iteration)
  
  submission = pd.DataFrame({
        "key": key,
        "fare_amount": prediction
  })

  submission.to_csv('submission.csv',index=False)

