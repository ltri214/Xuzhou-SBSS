# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LinearRegression
import mglearn
import mgtwr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from pygam import LinearGAM
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import scikitplot as skplt
from sklearn.metrics import roc_auc_score
import shap



file_name = '7.11D.csv'
data = pd.read_csv(file_name)

list1 =['pop', 'road_dis',
           'bus_dis', 'sub_dis', 'jun_dis',
          'busjoin',
          'shp_Large volume', 'shp_country', 'shp_crowd', 'shp_empty',
          'shp_road', 'shp_squire', 'fun_traffic', 'fun_admin', 'fun_business',
          'fun_infrastructure', 'fun_resident', 'fun_industry', 'fun_reserved', 'fun_nature',
          'shpstd', 'funstd', 'junction_num', 'length']

y = data['count']

# X = pd.read_csv('D_X.csv')
# X = X.fillna(0)
# remember = X
#
# for i in list1:
#     for j in X['name']:
#         link = j + '_distance'
#         a = i + '_' + j
#
#         X[a] = remember[i] * remember[link]
#     print('_'*20)
#
# X.to_csv('D_fin.csv',encoding='utf-8',index=None)

X = pd.read_csv('D_fin.csv')
for j in X['name']:
    link = j + '_distance'
    X.drop(link,axis=1,inplace=True)

X.drop('name',axis=1,inplace=True)

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=14,test_size=0.2)
# rf =  RandomForestRegressor(n_estimators=3000).fit(X_train,y_train)
#
# print("RF:")
# print("Training set score: {:.2f}".format(rf.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(rf.score(X_test, y_test)))
#
# a = np.array(rf.predict(X))
# a = pd.DataFrame(a)
#
# a.columns = ['rf_Ocount']
# a['ID'] = a.index+1
# a.to_csv('829_D.csv',index=False,encoding='utf-8')
rf =  RandomForestRegressor(n_estimators=3000)
score2 = -cross_val_score(rf,X=X,y=y,cv=5,scoring='neg_mean_absolute_error')
print('MAE')
print(score2)
print(score2.mean())
score = -cross_val_score(rf,X=X,y=y,cv=5,scoring='neg_root_mean_squared_error')
print('RMSE')
print(score)
print(score.mean())