print("\nImporting Libraries")
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from datetime import datetime

#from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
#import math

import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

#%% importing libraries
print("\nLoading the dataset")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

#%% Concat both the datasets, find X and y
print("\nPreprocessing")
y = train['count']
datetime_col = test['datetime']

train.drop(['casual', 'registered'],axis = 1, inplace = True)
data = pd.concat([train.iloc[:,:-1],test], axis = 0)

#%%
data["date"] = data.datetime.apply(lambda x : str(x).split()[0])
data["hour"] = data.datetime.apply(lambda x : (str(x).split()[1]).split(":")[0]).astype("int")
data["year"] = data.datetime.apply(lambda x : str(x).split()[0].split("-")[0])
data["weekday"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())
data["month"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)

#%% coercing To Category Type
print("\nSeparating categorical and numeric variables")
categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]
numericalFeatureNames = ["temp","humidity","windspeed"]

for var in categoricalFeatureNames:
    data[var] = data[var].astype("category")

data = pd.get_dummies(data, columns = ["season","weather","weekday","month","year","hour"])

#%% dropping unnecessary columns
    
dropFeatures = ["datetime","atemp","windspeed","date"]
data.drop(dropFeatures,axis=1, inplace = True)

#%% 
X = data.iloc[:len(train['count']),:]
test_df = data.iloc[len(train['count']):,:]

del data
#%%

y = np.log1p(y)
#%% train_test_split-ing
print("\nTrain Test Split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# def rmsle(y, y_,convertExp=True):
#     if convertExp:
#         y = np.exp(y),
#         y_ = np.exp(y_)
#     log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
#     log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
#     calc = (log1 - log2) ** 2
#     return np.sqrt(np.mean(calc))
# =============================================================================

print("\nFitting the model")
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X = X_train, y = y_train)
y_pred = rfr.predict(X = X_test)

from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import mean_squared_log_error

print("         MAE is ",mean_absolute_error(np.exp(y_test), np.exp(y_pred)))

#print("100xRMSLE is",100*math.sqrt(mean_squared_log_error(y_test, y_pred)))

#%%
#compare = pd.concat([pd.Series(y_test),pd.Series(y_pred)], keys = ['test','pred'], axis = 1)

#%% predict on test set
print("\nMaking Predictions")
y_pred_submission = rfr.predict(test_df)

#%% EXPORTING TO CSV
print("\nExporting to CSV")
submission = pd.DataFrame({"datetime": datetime_col, "count": np.expm1(y_pred_submission)})
# submission.to_csv('submissions/v4.csv', index=False)

# MAE dummy encoding without windspeed - 36.9069
# MAE OHE without windspeed - 33.0421 

print("\nDone")