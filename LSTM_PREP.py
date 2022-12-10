import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

#%%
# Read Train and Future Data

train_data = pd.read_csv("train_data.csv")
future_data = pd.read_csv("future_data.csv")

train_data["Date"] = pd.to_datetime(train_data["Date"],format="%Y-%m-%d")
future_data["Date"] = pd.to_datetime(future_data["Date"],format="%Y-%m-%d")

#%%
# Select Features to Use
features_to_use = ['Store','Promo','SchoolHoliday','CompetitionDistance','Month','WeekOfYear','Day',
                   'DayOfYear','Quarter','Competition_Effect','Affected',
                   'Prev_Quarter_Mdn','Sch_Holidays_Last_Wk',
                   'Promo2_Effect','Promo2_Type','Stat_Holidays','Avg_Sales_Per_Customer','Avg_Customers_While_Promo',
                   'Avg_Customers_While_Holiday','Avg_Days_Open','Avg_Promo_Per_Month',
                   'Avg_School_Holidays','Assortment','StoreType',
                   'DayOfWeek']

#%%
# Label Encoding Store

stores = train_data.copy()
stores = stores[["Store"]].drop_duplicates()
stores = stores.reset_index(drop=True)
stores = stores.reset_index()
stores = stores.rename(columns={"index":"Code"})

train_data = train_data.merge(stores,how="left",on=["Store"])
train_data = train_data.drop(["Store"],axis=1)
train_data = train_data.rename(columns={"Code":"Store"})

future_data = future_data.merge(stores,how="left",on=["Store"])
future_data = future_data.drop(["Store"],axis=1)
future_data = future_data.rename(columns={"Code":"Store"})

#%%
# Test-Train Split
# Last 3 Months for Validation

test_data = train_data.loc[train_data.Date>(train_data.Date.max() - pd.DateOffset(months=3))]
train_data = train_data.loc[train_data.Date<=(train_data.Date.max() - pd.DateOffset(months=3))]

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

#%%
# Scaling Data between 0 and 1

x_scaler = MinMaxScaler(feature_range=(0,1))
y_scaler = MinMaxScaler(feature_range=(0,1))

x_train = train_data[features_to_use]
x_test = test_data[features_to_use]

y_train = train_data[["Sales"]]
y_test = test_data[["Sales"]]

x_fut = future_data[features_to_use]

x_train_scaled = x_scaler.fit_transform(x_train)
x_train_scaled = pd.DataFrame(data=x_train_scaled,columns=x_train.columns)

x_test_scaled = x_scaler.transform(x_test)
x_test_scaled = pd.DataFrame(data=x_test_scaled,columns=x_test.columns)

x_fut_scaled = x_scaler.transform(x_fut)
x_fut_scaled = pd.DataFrame(data=x_fut_scaled,columns=x_fut.columns)

y_train_scaled = y_scaler.fit_transform(y_train)
y_train_scaled = pd.DataFrame(data=y_train_scaled,columns=y_train.columns)

y_test_scaled = y_scaler.transform(y_test)
y_test_scaled = pd.DataFrame(data=y_test_scaled,columns=y_test.columns)

train_scaled = pd.concat([x_train_scaled,y_train_scaled],axis=1)
test_scaled = pd.concat([x_test_scaled,y_test_scaled],axis=1)
future_scaled = x_fut_scaled
future_scaled["Sales"] = np.nan

#%%
# LSTM Look Back

timestep = 12

# Transforming Scaled Train Data for RNN Input
count = 0
for st in list(train_scaled.Store.unique()):
    temp = train_scaled[train_scaled.Store==st]
    temp = temp.reset_index(drop=True)

    empty_row = temp.copy().iloc[[0]]
    empty_row.iloc[[0]] = np.nan
    temp = pd.concat([temp,empty_row],axis=0)
    temp = temp.reset_index(drop=True)

    ts_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(temp.drop("Sales",axis=1).to_numpy(),temp[["Sales"]].shift(1).to_numpy(),length=timestep,batch_size=len(temp)-timestep)
    if count == 0:
        X_train = ts_gen[0][0]
        Y_train = ts_gen[0][1]
    else:
        X_train = np.concatenate([X_train,ts_gen[0][0]])
        Y_train = np.concatenate([Y_train,ts_gen[0][1]])
    count+=1

orig_train_df = pd.DataFrame()
for st in list(train_data.Store.unique()):
    temp = train_data[train_data.Store==st]
    temp = temp.reset_index(drop=True)
    orig_train_df = pd.concat([orig_train_df,temp.iloc[timestep-1:]])
orig_train_df = orig_train_df.reset_index(drop=True)

# Transforming Scaled Test Data for RNN Input
count = 0
for st in list(test_scaled.Store.unique()):
    temp = test_scaled[test_scaled.Store==st]
    temp = temp.reset_index(drop=True)

    empty_row = temp.copy().iloc[[0]]
    empty_row.iloc[[0]] = np.nan
    temp = pd.concat([temp,empty_row],axis=0)
    temp = temp.reset_index(drop=True)

    ts_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(temp.drop("Sales",axis=1).to_numpy(),temp[["Sales"]].shift(1).to_numpy(),length=timestep,batch_size=len(temp)-timestep)
    if count == 0:
        X_test = ts_gen[0][0]
        Y_test = ts_gen[0][1]
    else:
        X_test = np.concatenate([X_test,ts_gen[0][0]])
        Y_test = np.concatenate([Y_test,ts_gen[0][1]])
    count+=1

orig_test_df = pd.DataFrame()
for st in list(test_data.Store.unique()):
    temp = test_data[test_data.Store==st]
    temp = temp.reset_index(drop=True)
    orig_test_df = pd.concat([orig_test_df,temp.iloc[timestep-1:]])
orig_test_df = orig_test_df.reset_index(drop=True)

# Transforming Scaled Future Data for RNN Input
count = 0
for st in list(future_scaled.Store.unique()):
    temp = future_scaled[future_scaled.Store==st]
    temp = temp.reset_index(drop=True)

    temp = temp.iloc[12-timestep:]
    temp = temp.reset_index(drop=True)

    empty_row = temp.copy().iloc[[0]]
    empty_row.iloc[[0]] = np.nan
    temp = pd.concat([temp,empty_row],axis=0)
    temp = temp.reset_index(drop=True)

    ts_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(temp.drop("Sales",axis=1).to_numpy(),temp[["Sales"]].shift(1).to_numpy(),length=timestep,batch_size=len(temp)-timestep)
    if count == 0:
        X_fut = ts_gen[0][0]
    else:
        X_fut = np.concatenate([X_fut,ts_gen[0][0]])
    count+=1

orig_future_df = pd.DataFrame()
for st in list(future_data.Store.unique()):
    temp = future_data[future_data.Store==st]
    temp = temp.reset_index(drop=True)

    temp = temp.iloc[12-timestep:]
    temp = temp.reset_index(drop=True)

    orig_future_df = pd.concat([orig_future_df,temp.iloc[timestep-1:]])
orig_future_df = orig_future_df.reset_index(drop=True)

#%%
# Saving Files

joblib.dump(x_scaler,'xscaler.save')
joblib.dump(y_scaler,'yscaler.save')

np.save('X_train',X_train)
np.save('Y_train',Y_train)
orig_train_df.to_csv("orig_train_df.csv",index=False)

np.save('X_test',X_test)
np.save('Y_test',Y_test)
orig_test_df.to_csv("orig_test_df.csv",index=False)

np.save('X_fut',X_fut)
orig_future_df.to_csv("orig_future_df.csv",index=False)

stores.to_csv("Store_Labels.csv",index=False)

print("X_Train Shape = ",X_train.shape)
print("X_Test Shape = ",X_test.shape)
print("X_fut Shape = ",X_fut.shape)
print("Y_Train Shape = ",Y_train.shape)
print("Y_Test Shape = ",Y_test.shape)

print("Train Shape = ",orig_train_df.shape)
print("Test Shape = ",orig_test_df.shape)
print("fut Shape = ",orig_future_df.shape)




