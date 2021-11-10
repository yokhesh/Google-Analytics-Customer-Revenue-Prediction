#### Code for loading data and unpacking JSON columns
import numpy as np 
import pandas as pd 
import os
import json
from pandas.io.json import json_normalize
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 
import seaborn as sns

neccolu = ['device', 'geoNetwork', 'totals', 'trafficSource']

train = pd.read_csv('../input/train_v2.csv', converters={column:json.loads for column in neccolu})
test = pd.read_csv('../input/test_v2.csv', converters={column:json.loads for column in neccolu})
train = train.drop('hits',axis = 1)
test = test.drop('hits',axis = 1)

### Function to unpack the JSON columns
def jsounpac(neccolu,data):
    for columns in neccolu:
        column_as_df = json_normalize(data[columns])
        data = data.drop(columns, axis=1)
        data = pd.concat([data, column_as_df], axis=1)
    return data

train_unpack = jsounpac(neccolu,train)
del train

test_unpack = jsounpac(neccolu,test)
del test

###Removing column with demo dataset####
def remove_demo(u):
    demo_col = 0
    for col in u.columns:
        if col == "hits":
            continue
        t = u.groupby(col).size().reset_index(name='count')
        r =  str(t[col]).split(' ')
        s = str((((str(t['count']).split(' '))[4]).split('\n'))[0])
        #if r[7] == 'demo' and s == '10000':
        if r[7] == 'demo':
            demo_col = demo_col +1
            u = u.drop(col, axis=1)
    return u

train_no_demo = remove_demo(train_unpack)

test_no_demo = remove_demo(test_unpack)

#####Creating Correlation matrix for the dataset
corr = train_no_demo.corr()
corr.style.background_gradient(cmap='coolwarm')

##removing columns with least correlation values with the target value
###removing columns with most values as demo
train_no_demo = train_no_demo.drop('city',axis = 1)
train_no_demo = train_no_demo.drop('metro',axis = 1)
train_no_demo = train_no_demo.drop('region',axis = 1)
train_no_demo = train_no_demo.drop('hits',axis = 1)
train_no_demo = train_no_demo.drop('adwordsClickInfo.gclId',axis = 1)
train_no_demo = train_no_demo.drop('adContent',axis = 1)
train_no_demo = train_no_demo.drop('adwordsClickInfo.adNetworkType',axis = 1)
train_no_demo = train_no_demo.drop('adwordsClickInfo.isVideoAd',axis = 1)
train_no_demo = train_no_demo.drop('adwordsClickInfo.page',axis = 1)
train_no_demo = train_no_demo.drop('adwordsClickInfo.slot',axis = 1)
train_no_demo = train_no_demo.drop('keyword',axis = 1)
train_no_demo = train_no_demo.drop('referralPath',axis = 1)
train_no_demo = train_no_demo.drop('visitId',axis = 1)

####Visualizations
###Transactions per country
ry = train_no_demo[['continent','transactionRevenue']]
ry['transactionRevenue'] = (ry['transactionRevenue'].fillna(0))
ry['transactionRevenue'][ry['transactionRevenue'] != 0] = 1
trans_per_country = ry.groupby(['continent','transactionRevenue']).size().reset_index(name='count')

##people who have actually spent
no_spent = train_no_demo.transactionRevenue.isna().sum()
spent =  len(train_no_demo) - no_spent
trans_gr = pd.DataFrame()
trans_gr = pd.DataFrame({'Heading':['No transactions done','Transactions done'], 'Count':[no_spent,spent]})
plt.figure(figsize=(10, 5))
plt.bar(trans_gr['Heading'], trans_gr['Count'], width=0.3) 
plt.xlabel('Category of visitors')
plt.ylabel('Frequency of transaction')
plt.title('Number of visitors who have made a transaction')



###Number of browsers in each device

brow_device = train_no_demo.groupby(['deviceCategory','browser']).size().reset_index(name='count')
desktop = brow_device.iloc[0:15,:]
mobile = brow_device.iloc[15:30,:]
tablet = brow_device.iloc[30:38,:]
plt.figure(figsize=(60, 20))
plt.bar(desktop['browser'], desktop['count'], width=0.5, label='Desktop') 
plt.bar(mobile['browser'], mobile['count'], width=0.5, label='Mobile') 
plt.bar(tablet['browser'], tablet['count'], width=0.5, label='Tablet') 
plt.legend()
plt.xlabel('Different browser per device')
plt.ylabel('Frequency of browser used')
plt.title('Number of times a browser is used in a device')

###No of visits for a unique ID
unique_id = train_no_demo.groupby('fullVisitorId').size().reset_index(name='count')
unique_id1 = unique_id.groupby('count').size().reset_index(name='count1')
plt.figure(figsize=(10, 5))
plt.bar(unique_id1['count'], unique_id1['count1'], width=0.3) 
plt.xlabel('Number of visits per unique visitor')
plt.ylabel('Number of unique visitors')
plt.title('Visits per unique visitor')

##replacing the nan values
##Numeric values
train_no_demo['transactionRevenue'] = (train_no_demo['transactionRevenue'].fillna(0))
train_no_demo['transactions'] = (train_no_demo['transactions'].fillna(0))
train_no_demo['totalTransactionRevenue'] = (train_no_demo['totalTransactionRevenue'].fillna(0))
train_no_demo['bounces'] = (train_no_demo['bounces'].fillna(0))
train_no_demo['timeOnSite'] = (train_no_demo['timeOnSite'].fillna(0))
train_no_demo['sessionQualityDim'] = (train_no_demo['sessionQualityDim'].fillna(0))
train_no_demo['newVisits'] = (train_no_demo['newVisits'].fillna(0))
train_no_demo['pageviews'] = (train_no_demo['pageviews'].fillna(0))
##Boolean values
train_no_demo['isTrueDirect'] = (train_no_demo['isTrueDirect'].fillna(False))
##Finding log values for the target value
train_no_demo['transactionRevenue'] = np.log1p(train_no_demo['transactionRevenue'].astype(float))

####Statistical plots for the transaction value
tes = pd.DataFrame()
tes = pd.DataFrame({'transaction_revenue':train_no_demo['transactionRevenue']})
tes = tes[(tes != 0).all(1)]
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(10)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

tes.plot.box(ax= ax1)
ax1.set_title("Box plot for the non-zero transactions")
tes.plot.kde(ax= ax2)
ax2.set_xlabel('transaction value')
ax2.set_title("Kde plot for the non-zero transactions")
tes.hist(ax= ax3)
ax3.set_xlabel('transaction value')
ax3.set_ylabel('Frequency')
ax3.set_title("Histogram for the non-zero transactions")
tes.plot.area(ax= ax4)
ax4.set_title("Area plot for the non-zero transactions")


#####Type conversion

train_no_demo['pageviews'] = train_no_demo['pageviews'].astype(int)
train_no_demo['newVisits'] = train_no_demo['newVisits'].astype(int)
train_no_demo['bounces'] = train_no_demo['bounces'].astype(int)

####Converting categorical values to numeric values
for i, t in train_no_demo.loc[:, train_no_demo.columns != 'fullVisitorId'].dtypes.iteritems():
    if t == object:
        train_no_demo[i].fillna('unknown', inplace=True)
        train_no_demo[i] = pd.factorize(train_no_demo[i])[0]

#### Checking if the dataset has any NAN values
null_cnt = train_no_demo.isnull().sum().sort_values()
print(null_cnt[null_cnt > 0])

####Splitting the training and validation data

from sklearn.model_selection import train_test_split
trn_x, val_x, trn_y, val_y = train_test_split(train_no_demo, y_train, test_size=0.2, random_state=None)

#########Regression Models###########

#####Deep Neural Networks

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Dropout
from keras import optimizers
col = [coli for coli in trn_x.columns]
scaler = preprocessing.MinMaxScaler()
trn_x[col] = scaler.fit_transform(trn_x[col])
val_x[col] = scaler.transform(val_x[col])
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0003
model = Sequential()
model.add(Dense(256, kernel_initializer='glorot_normal', activation='relu', input_dim=trn_x.shape[1]))
model.add(Dense(128, kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(64, kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(32, kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(1))
adam = optimizers.adam(lr=LEARNING_RATE)
model.compile(loss='mse', optimizer=adam)
model.fit(x=trn_x.values, y=trn_y, epochs=10, verbose=2, validation_data=(val_x.values, val_y))
y_neu = model.predict(val_x)

#########Decision Tree

from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf = clf.fit(trn_x, trn_y)
y_dec = clf.predict(val_x)

##########Linear Regression

from sklearn import linear_model
reg = linear_model.BayesianRidge()
reg.fit(trn_x, trn_y)
y_li = reg.predict(val_x)


#########Random Forest

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
regr.fit(trn_x, trn_y)
y_ran = regr.predict(val_x)

###########LGBM

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
params={'learning_rate': 0.02,
        'objective':'regression',
        'metric':'rmse',
        'num_leaves': 31,
        'verbose': 1,
        'random_state':42,
        'bagging_fraction': 0.6,
        'feature_fraction': 0.6
       }
folds = GroupKFold(n_splits=5)

reg = lgb.LGBMRegressor(**params, n_estimators=3000)
reg.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=100, verbose=500)
y_lgbm = reg.predict(val_x, num_iteration=reg.best_iteration_)

############XGB

from xgboost import XGBRegressor
xgb_params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.02,
        'max_depth': 22,
        'min_child_weight': 57,
        'gamma' : 1.45,
        'alpha': 0.0,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 456
    }
xg = XGBRegressor(**xgb_params, n_estimators=3000)
xg.fit(trn_x, trn_y,eval_set=[(val_x, val_y)],early_stopping_rounds=50,eval_metric='rmse',verbose=100)
y_xgb = xg.predict(val_x)

############Function for calculating the performance metrics for each of the regression models

def metrics(val_y,y_pred):
    diff = pd.DataFrame()
    diff['val_true'] = val_y
    diff['val_pred'] = y_pred
    vs = explained_variance_score(diff.val_true, diff.val_pred)
    mae = mean_absolute_error(diff.val_true, diff.val_pred)
    mse =mean_squared_error(diff.val_true, diff.val_pred)
    meae = median_absolute_error(diff.val_true, diff.val_pred)
    r2 = r2_score(diff.val_true, diff.val_pred)
    result = pd.DataFrame({'error':[vs,mae,mse,meae,r2]})
    return result


###########Attaching prediction results for all the models into one dataframe for calculating the metrics
diff_error = pd.DataFrame()
diff_error['DNN'] =y_neu[:,0]
diff_error['xgb'] = y_xgb
diff_error['lgbm'] = y_lgbm
diff_error['rand_forest'] = y_ran
diff_error['linear'] = y_li
diff_error['decision'] = y_dec

##############Calculating the metrics
##############The variable ‘ovr_result’ contains the metrics table

ovr_result= pd.DataFrame()
ovr_result = pd.DataFrame({'Evaluation Metrics':['Explained Variance Score','Mean Absolute Error','Mean Squared Error','Median Absolute Error','R2 Score']})
for column in diff_error:
    f=diff_error[column].as_matrix()
    result = metrics(val_y,f)
    ovr_result[column] = result['error'].tolist()

##############Plotting the predicted values

diff_pl = []
diff_pl = diff_error 
diff_pl['true_val'] = val_y.as_matrix()
diff_pl = diff_pl[diff_pl.true_val != 0]
diff_pl.insert(0, 'New_ID1', range(1, 1 + len(diff_pl)))

re = diff_pl.head(200)

plt.figure(figsize=(8, 3))
plt.plot(re.New_ID1,re.DNN,'r', label='Deep Neural Networks')
plt.plot(re.New_ID1,re.lgbm,'g', label='LGBM')
plt.plot(re.New_ID1,re.xgb,'k', label='XGBOOST')
plt.plot(re.New_ID1,re.linear,'y', label='Linear Regressor', linewidth=0.5)
plt.plot(re.New_ID1,re.rand_forest,'k', label='Random Forest', linewidth=0.5)
plt.plot(re.New_ID1,re.decision,'g', label='Decision Tree')
plt.plot(re.New_ID1,re.true_val,'b', label='True Value', linewidth=0.3)
plt.legend()
plt.xlabel('Instances')
plt.ylabel('Target value')
plt.title('Comparison of multiple regressors and the true value for 200 instances')
plt.show()
