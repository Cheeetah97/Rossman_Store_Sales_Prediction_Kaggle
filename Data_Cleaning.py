import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#%%
# Ready all the relevant Data Sets

path = "E:/Masters/NN_and_Deep_Learning/Group_Project/Data/"

train_data = pd.read_csv(path+"train.csv")
test_data = pd.read_csv(path+"test.csv") 
store_data = pd.read_csv(path+"store.csv")

#%%
# Filling NaN Values in "CompetitionDistance" field and Merging with the Train Data Set
# Creating Time Series Features from the Date Column

store_data["CompetitionDistance"] = store_data["CompetitionDistance"].fillna(store_data["CompetitionDistance"].mean())

train_data = train_data.merge(store_data,on=["Store"],how='left')

train_data["Date"] = pd.to_datetime(train_data["Date"],format="%Y-%m-%d")
train_data["Year"] = train_data["Date"].dt.year
train_data["Month"] = train_data["Date"].dt.month
train_data["WeekOfYear"] = train_data["Date"].dt.week
train_data["Day"] = train_data["Date"].dt.day
train_data["DayOfYear"] = train_data["Date"].dt.dayofyear
train_data["Quarter"] = train_data["Date"].dt.quarter
    
#%%
# Remove Zeros from Start of Each Store's Timeseries and the Outliers
# Outliers are defined as data points that lie outside a certain range which in this case
# is greater than 3.5 Standard Deviation from the Trend Line for High Falling Points
# is less than 2.5 Standard Deviation from the Trend Line
# These ranges were selected after Visualizing the data before and after Outlier Removal

df = pd.DataFrame()
for st in list(train_data.Store.unique()):
    temp = train_data[train_data.Store==st]
    temp = temp.sort_values(by=["Date"])
    temp = temp.reset_index(drop=True)
    
    if temp.loc[temp.Open!=0,"Open"].index.to_series().idxmin()>=2:
        temp = temp.iloc[temp.loc[temp.Open!=0,"Open"].index.to_series().idxmin():]
        temp = temp.reset_index(drop=True)
    
    temp_yr_df = pd.DataFrame()
    for yr in list(temp.Year.unique()):
        temp_yr = temp.loc[(temp.Year==yr)&(temp.Open==1)]
        model = LinearRegression()
        X = [i for i in range(1,len(temp_yr)+1,1)]
        X = np.reshape(X,(len(X),1))
        Y = temp_yr["Sales"].values
        model.fit(X,Y)
        temp_yr["t_line"] = model.predict(X)
        temp_yr_df = pd.concat([temp_yr_df,temp_yr[["Date","t_line"]]])
    
    temp = temp.merge(temp_yr_df,how="left",on=["Date"])
    st_std = temp.loc[temp.Open==1,"Sales"].std()
    
    temp.loc[(temp.Sales>(temp.t_line+3.5*st_std))&(temp.Month!=12)&(temp.Open==1),"Sales"] = temp.loc[(temp.Sales>(temp.t_line+3.5*st_std))&(temp.Month!=12)&(temp.Open==1),"t_line"] + 2*st_std
    temp.loc[(temp.Sales<(temp.t_line-2.5*st_std))&(temp.Open==1)&(temp.DayOfWeek!=7),"Sales"] = temp.loc[(temp.Sales<(temp.t_line-2.5*st_std))&(temp.Open==1)&(temp.DayOfWeek!=7),"t_line"] - 1.5*st_std
    
    df = pd.concat([df,temp])
       
df.to_csv(path+"Cleaned_data.csv",index=False)

