import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import r2_score

#%%
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
print(train.head())

#%%
print(train.shape)

# train set has 10886 rows and 12 columns

print(train.head(2))

#datetime       object
#season          int64
#holiday         int64
#workingday      int64
#weather         int64
#temp          float64
#atemp         float64
#humidity        int64
#windspeed     float64
#casual          int64
#registered      int64
#count           int64
#dtype: object

print(train.dtypes)
#%%
def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


#%%
train  = train.drop(["datetime"],axis=1)

#%% CORELATION ANALYSIS

corrMatt = train[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)

#%%
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)
fig.set_size_inches(12, 5)
sns.regplot(x="temp", y="count", data=train, ax=ax1)
sns.regplot(x="windspeed", y="count", data=train, ax=ax2)
sns.regplot(x="humidity", y="count", data=train, ax=ax3)

#%%

y = train['count']
train.drop(['casual', 'registered'],axis = 1, inplace = True)
data = pd.concat([train.iloc[:,:-1],test], axis = 0)

#%%
data["date"] = data.datetime.apply(lambda x : str(x).split()[0])
data["hour"] = data.datetime.apply(lambda x : (str(x).split()[1]).split(":")[0]).astype("int")
data["year"] = data.datetime.apply(lambda x : str(x).split()[0].split("-")[0])
data["weekday"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())
data["month"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)

#%% Random Forest Model To Predict 0's In Windspeed

# from sklearn.ensemble import RandomForestRegressor

# dataWind0 = data[data["windspeed"]==0]
# dataWindNot0 = data[data["windspeed"]!=0]
# rfModel_wind = RandomForestRegressor()
# windColumns = ["season","weather","humidity","month","temp","year","atemp"]
# rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])

# wind0Values = rfModel_wind.predict(X= dataWind0[windColumns])
# dataWind0["windspeed"] = wind0Values
# data = dataWindNot0.append(dataWind0)
# data.reset_index(inplace=True)
# data.drop('index',inplace=True,axis=1)


#%%
# Coercing To Category Type

categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]
numericalFeatureNames = ["temp","humidity","windspeed","atemp"]
dropFeatures = ['casual',"count","datetime","date","registered"]


for var in categoricalFeatureNames:
    data[var] = data[var].astype("category")
    
    

#%% TRAIN TEST SPLIT
    
dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])
dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"])
datetimecol = dataTest["datetime"]
yLabels = dataTrain["count"]
yLablesRegistered = dataTrain["registered"]
yLablesCasual = dataTrain["casual"]

#%%
dataTrain  = dataTrain.drop(dropFeatures,axis=1)
dataTest  = dataTest.drop(dropFeatures,axis=1)

#%% RMSLE SCORER

def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

#%% LINEAR MODEL

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logistic regression model
lModel = LinearRegression()

# dataTrain the model
yLabelsLog = np.log1p(yLabels)
lModel.fit(X = dataTrain, y = yLabelsLog)

# Make predictions
preds = lModel.predict(X = dataTrain)
print ("RMSLE Value For Linear Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))

# LOT OF NAN VALUES AFTER 9570 IN BOTH ACTUAL AND PREDICTED 
print("R-squared for Linear Regression: ",r2_score(y_true = yLabels[:9570], y_pred = np.exp(preds[:9570])))

# y_true = np.exp(yLabelsLog)

#%% CREATE DATAFRAME WIH ACTUAL AND PREDICTED VALUES

cols = ['actual', 'predicted']
del3 = pd.concat([pd.Series(yLabels),pd.Series(np.exp(preds))],axis = 1)
del3.columns = cols
#del1 = pd.DataFrame([yLabels,np.exp(preds)],columns = cols)

#%% EXPORTING TO CSV

submission = pd.DataFrame({
        "datetime": datetimecol,
        "count": [max(0, x) for x in np.exp(predsTest)]
    })

submission.to_csv('bike_predictions_gbm_separate_without_fe.csv', index=False)



