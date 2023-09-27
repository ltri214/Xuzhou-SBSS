# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import shap

file_name = 'O.csv'
data = pd.read_csv(file_name)
print(data.columns)

data = data.dropna()
data= data[['count','chn_ppp_20','road_dis',
           'busdis', 'subdis', 'jun_dis','busjoin',
    'shp_Large volume', 'shp_country', 'shp_crowd', 'shp_empty',
       'shp_road', 'shp_squire', 'fun_traffic', 'fun_admin', 'fun_bussiness',
       'fun_infrastructure', 'fun_resident', 'fun_industry', 'fun_reserved', 'fun_nature',
       'shpstd', 'funstd',
            'junction_num', 'length','bw_1','bw_2','bw=3','bw=4','bw=5','bw=6','bw=7','bw=8']]


X = data[['chn_ppp_20','road_dis',
           'busdis', 'subdis', 'jun_dis','busjoin',
    'shp_Large volume', 'shp_country', 'shp_crowd', 'shp_empty',
       'shp_road', 'shp_squire', 'fun_traffic', 'fun_admin', 'fun_bussiness',
       'fun_infrastructure', 'fun_resident', 'fun_industry', 'fun_reserved', 'fun_nature',
       'shpstd', 'funstd',
            'junction_num', 'length','bw=3']]

X = X.fillna(0)

y = data['count']


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=14,test_size=0.2)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

linner = LinearRegression().fit(X,y)

rf = RandomForestRegressor(n_estimators=3000).fit(X_train,y_train)

xg = xgb.XGBRegressor(n_estimators=50000,max_depth = 4, learning_rate = 0.00005,objective = 'reg:squarederror'
                      ,booster='gbtree').fit(X_train,y_train)

gbm = lgb.LGBMRegressor(objective='regression',learning_rate=0.03,n_estimators=50000)
gbm.fit(X_train,y_train,eval_set=[(X_test,y_test)], eval_metric='l1', early_stopping_rounds=5)



# score = -cross_val_score(rf,X=X,y=y,cv=10,scoring='r2')
# score3 = -cross_val_score(xg,X=X,y=y,cv=10,scoring='neg_mean_absolute_error')
# score4 = -cross_val_score(gbm,X=X,y=y,cv=10,scoring='neg_mean_absolute_error')



# print(score)

# print("RF:")
# print(score.mean())
# print("xg:")
# print(score3.mean())
# print("gbm:")
# print(score4.mean())

# score = -cross_val_score(rf,X=X,y=y,cv=10,scoring='neg_root_mean_squared_error')
# score3 = -cross_val_score(xg,X=X,y=y,cv=10,scoring='neg_root_mean_squared_error')
# score4 = -cross_val_score(gbm,X=X,y=y,cv=10,scoring='neg_root_mean_squared_error')
# print("RF:")
# print(score.mean())
# print("xg:")
# print(score3.mean())
# print("gbm:")
# print(score4.mean())




# print(result)
print("LR:")
print("Training set score: {:.2f}".format(linner.score(X_train, y_train)))
print("Test set score: {:.2f}".format(linner.score(X_test, y_test)))
print("RF:")
print("Training set score: {:.2f}".format(rf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(rf.score(X_test, y_test)))
print("xg:")
print("Training set score: {:.2f}".format(xg.score(X_train, y_train)))
print("Test set score: {:.2f}".format(xg.score(X_test, y_test)))
print("gbm:")
print("Training set score: {:.2f}".format(gbm.score(X_train, y_train)))
print("Test set score: {:.2f}".format(gbm.score(X_test, y_test)))
