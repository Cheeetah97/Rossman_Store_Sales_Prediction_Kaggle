import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#%%
# Reading Cleaned Data from the previously generated file

path = "E:/Masters/NN_and_Deep_Learning/Group_Project/Data/"

train_data = pd.read_csv(path+"Cleaned_Data.csv")
future_data = pd.read_csv(path+"test.csv") 
comp_data_blanks = pd.read_csv(path+"CompetitionDataBlanks.csv")

train_data["Date"] = pd.to_datetime(train_data["Date"],format="%Y-%m-%d")
future_data["Date"] = pd.to_datetime(future_data["Date"],format="%Y-%m-%d")

#%%
# Basic Store Information
# Number of Stores

stores_test = future_data["Store"].unique()
stores_train = train_data["Store"].unique()

print("Total Stores in Test Data = ",future_data["Store"].nunique())
print("Total Stores in Train Data = ",train_data["Store"].nunique())
print("Stores in test but not in train = ",future_data.loc[~(future_data.Store.isin(stores_train)),"Store"].nunique())
print("Stores in train but not in test = ",train_data.loc[~(train_data.Store.isin(stores_test)),"Store"].nunique())
#%%
# Finding Stores affected by Competitor Launch and Making a Feature for them

df = train_data.loc[(train_data["CompetitionOpenSinceYear"]>=2013)&(train_data["CompetitionOpenSinceYear"]<=2015)]
empty = train_data.loc[(train_data["CompetitionOpenSinceYear"].isnull())|(train_data["CompetitionOpenSinceYear"].isnull())]
constant = train_data.loc[(train_data["CompetitionOpenSinceYear"]<2013)|(train_data["CompetitionOpenSinceYear"]>2015)]
print("Stores with Competitions Opened in 2013-2015 = ",df.Store.nunique())
print("Stores with Null Competition Open Date = ",empty.Store.nunique())
print("Stores with Competitions opened before/after 2013-2015 = ",constant.Store.nunique())

df = pd.DataFrame()
for st in list(train_data.Store.unique()):
    temp = train_data[train_data.Store==st]
    temp = temp.sort_values(by=["Date"])
    temp = temp.reset_index(drop=True)
    
    if st in list(comp_data_blanks.Store.unique()):
        temp_st = comp_data_blanks.loc[comp_data_blanks.Store==st]
        temp_st = temp_st.reset_index(drop=True)
        if temp_st["Affected"].iloc[0] == 1:
            temp["CO_Date"] = pd.to_datetime(temp_st["CompetitionOpenSinceYear"].iloc[0].astype(int).astype(str)+"-"+temp_st["CompetitionOpenSinceMonth"].iloc[0].astype(int).astype(str)+"-"+"01",format="%Y-%m-%d")
            
            temp2 = temp.loc[~(temp.Month.isin([12,4]))]
            temp2 = temp2.loc[temp2.Open==1]
            temp_after = temp2.loc[(temp2.Date>temp2.CO_Date)&(temp2.Date<=temp2.CO_Date+pd.DateOffset(months=4))]
            
            if len(temp_after) == 0:
                temp_after = temp2.loc[(temp2.Date>temp2.CO_Date)]
                temp_before = temp2.loc[(temp2.Date<=temp2.CO_Date)]
            else:
                temp_before = temp2.loc[(temp2.Date>temp2.CO_Date-pd.DateOffset(days=len(temp_after)))&(temp2.Date<=temp2.CO_Date)]
                if len(temp_before) < 30:
                    temp_after = temp2.loc[(temp2.Date>temp2.CO_Date)]
                    temp_before = temp2.loc[(temp2.Date<=temp2.CO_Date)]
            
            new_mean = temp_after["Sales"].mean()
            old_mean = temp_before["Sales"].mean()
            
            temp["Competition_Effect"] = 0
            if new_mean <= old_mean:
                temp.loc[(temp.Date>temp.CO_Date),"Competition_Effect"] = -(new_mean/old_mean)
            else:
                temp.loc[(temp.Date>temp.CO_Date),"Competition_Effect"] = old_mean/new_mean
            temp["Affected"] = 1
            df = pd.concat([df,temp])
        else:
            temp["Competition_Effect"] = 0
            temp["Affected"] = 0
            df = pd.concat([df,temp])
    else:
        temp["CO_Date"] = pd.to_datetime(temp["CompetitionOpenSinceYear"].astype(int).astype(str)+"-"+temp["CompetitionOpenSinceMonth"].astype(int).astype(str)+"-"+"01",format="%Y-%m-%d")
        
        if (temp["CO_Date"].iloc[0]< temp["Date"].min()+pd.DateOffset(months=3)):
            temp["Competition_Effect"] = 0
            temp["Affected"] = 0
            df = pd.concat([df,temp])
            continue
        
        temp2 = temp.loc[~(temp.Month.isin([12,4]))]
        temp2 = temp2.loc[temp2.Open==1]
        temp_after = temp2.loc[(temp2.Date>temp2.CO_Date)&(temp2.Date<=temp2.CO_Date+pd.DateOffset(months=4))]
        
        if len(temp_after) == 0:
            temp_after = temp2.loc[(temp2.Date>temp2.CO_Date)]
            temp_before = temp2.loc[(temp2.Date<=temp2.CO_Date)]
        else:
            temp_before = temp2.loc[(temp2.Date>temp2.CO_Date-pd.DateOffset(days=len(temp_after)))&(temp2.Date<=temp2.CO_Date)]
            if len(temp_before) < 30:
                temp_after = temp2.loc[(temp2.Date>temp2.CO_Date)]
                temp_before = temp2.loc[(temp2.Date<=temp2.CO_Date)]
                
        new_mean = temp_after["Sales"].mean()
        old_mean = temp_before["Sales"].mean()
        
        if st in [269,550,496,595,718,1053]:
            temp["Competition_Effect"] = 0
            if new_mean <= old_mean:
                temp.loc[(temp.Date>temp.CO_Date),"Competition_Effect"] = -(new_mean/old_mean)
            else:
                temp.loc[(temp.Date>temp.CO_Date),"Competition_Effect"] = old_mean/new_mean
            temp["Affected"] = 1
            df = pd.concat([df,temp])
        elif st in [225,326,859,996,1099,1085,1045]:
            temp["Competition_Effect"] = 0
            if new_mean <= old_mean:
                temp.loc[(temp.Date>temp.CO_Date),"Competition_Effect"] = -(new_mean/old_mean)
            else:
                temp.loc[(temp.Date>temp.CO_Date),"Competition_Effect"] = old_mean/new_mean
            temp["Affected"] = 0
            df = pd.concat([df,temp])
        elif (new_mean > old_mean-(0.25*temp_before["Sales"].std())) & (new_mean < old_mean+(0.25*temp_before["Sales"].std())):
            temp["Competition_Effect"] = 0
            if new_mean <= old_mean:
                temp.loc[(temp.Date>temp.CO_Date),"Competition_Effect"] = -(new_mean/old_mean)
            else:
                temp.loc[(temp.Date>temp.CO_Date),"Competition_Effect"] = old_mean/new_mean
            temp["Affected"] = 0
            df = pd.concat([df,temp])
        else:
            temp["Competition_Effect"] = 0
            if new_mean <= old_mean:
                temp.loc[(temp.Date>temp.CO_Date),"Competition_Effect"] = -(new_mean/old_mean)
            else:
                temp.loc[(temp.Date>temp.CO_Date),"Competition_Effect"] = old_mean/new_mean
            temp["Affected"] = 1
            df = pd.concat([df,temp])

train_data = df.copy()
train_data = train_data.reset_index(drop=True)

#%%
# Feature for Promo2 and Promo2 Interval

m = {"Jan":(12,1),
     "Feb":(1,2),
     "Mar":(2,3),
     "Apr":(3,4),
     "May":(4,5),
     "Jun":(5,6),
     "Jul":(6,7),
     "Aug":(7,8),
     "Sept":(8,9),
     "Oct":(9,10),
     "Nov":(10,11),
     "Dec":(11,12)}

df = pd.DataFrame()
for st in list(train_data.Store.unique()):
    temp = train_data[train_data.Store==st]
    temp = temp.sort_values(by=["Date"])
    temp = temp.reset_index(drop=True)
    
    # Median of Previous Quarter where Promo is 1
    temp["Prev_Quarter_Mdn"] = 0
    for yr in list(temp.Year.unique()):
        temp_yr = temp[temp.Year==yr]
        for qt in list(temp_yr.Quarter.unique()):
            if (yr==2013) & (qt==1):
                continue
            else:
                if qt == 1:
                    yr_prev = yr - 1
                    qt_prev = 4
                else:
                    yr_prev = yr
                    qt_prev = qt - 1
                temp.loc[(temp.Year==yr)&(temp.Quarter==qt),"Prev_Quarter_Mdn"] = temp.loc[(temp.Year==yr_prev)&
                                                                                           (temp.Quarter==qt_prev)&
                                                                                           (temp.Promo == 1),"Sales"].median()      
    temp["Prev_Quarter_Mdn"] = temp["Prev_Quarter_Mdn"].ffill()
    temp["Prev_Quarter_Mdn"] = temp["Prev_Quarter_Mdn"].bfill()
    
    # School Holidays Last Week and Same Month,Quarter and Week of Last Year
    temp["Sch_Holidays_Last_Wk"] = 0
    temp["Prev_YrQuarter_Mdn"] = 0
    temp["Prev_YrMonth_Mdn"] = 0
    temp["Prev_YrWeek_Mdn"] = 0
    
    for k,v in temp.iterrows():
        
        if k > 6:
            temp["Sch_Holidays_Last_Wk"].iloc[k] = temp["SchoolHoliday"].iloc[k-7:k].mean() 
            
        if temp["Year"].iloc[k] == 2014:
            
            if len(temp.loc[(temp.Year==2013)&(temp.Month==temp.Month.iloc[k])]) > 0:
                temp["Prev_YrMonth_Mdn"].iloc[k] = temp.loc[(temp.Year==2013)&(temp.Month==temp.Month.iloc[k]),"Sales"].median()
            else:
                temp["Prev_YrMonth_Mdn"].iloc[k] = temp.loc[(temp.Year==2013),"Sales"].median()   
            if len(temp.loc[(temp.Year==2013)&(temp.Quarter==temp.Quarter.iloc[k])]) > 0:
                temp["Prev_YrQuarter_Mdn"].iloc[k] = temp.loc[(temp.Year==2013)&(temp.Quarter==temp.Quarter.iloc[k]),"Sales"].median()
            else:
                temp["Prev_YrQuarter_Mdn"].iloc[k] = temp.loc[(temp.Year==2013),"Sales"].median()
            if len(temp.loc[(temp.Year==2013)&(temp.WeekOfYear==temp.WeekOfYear.iloc[k])]) > 0:
                temp["Prev_YrWeek_Mdn"].iloc[k] = temp.loc[(temp.Year==2013)&(temp.WeekOfYear==temp.WeekOfYear.iloc[k]),"Sales"].median()
            else:
                temp["Prev_YrWeek_Mdn"].iloc[k] = temp.loc[(temp.Year==2013),"Sales"].median()
                
                
        elif temp["Year"].iloc[k] == 2015:
            if len(temp.loc[(temp.Year==2014)&(temp.Month==temp.Month.iloc[k])]) > 0:
                temp["Prev_YrMonth_Mdn"].iloc[k] = temp.loc[(temp.Year==2014)&(temp.Month==temp.Month.iloc[k]),"Sales"].median()
            elif len(temp.loc[(temp.Year==2013)&(temp.Month==temp.Month.iloc[k])]) > 0:
                temp["Prev_YrMonth_Mdn"].iloc[k] = temp.loc[(temp.Year==2013)&(temp.Month==temp.Month.iloc[k]),"Sales"].median() 
            else:
                temp["Prev_YrMonth_Mdn"].iloc[k] = temp.loc[(temp.Year==2014),"Sales"].median()   
                
            if len(temp.loc[(temp.Year==2014)&(temp.Quarter==temp.Quarter.iloc[k])]) > 0:
                temp["Prev_YrQuarter_Mdn"].iloc[k] = temp.loc[(temp.Year==2014)&(temp.Quarter==temp.Quarter.iloc[k]),"Sales"].median()
            elif len(temp.loc[(temp.Year==2013)&(temp.Quarter==temp.Quarter.iloc[k])]) > 0:
                temp["Prev_YrQuarter_Mdn"].iloc[k] = temp.loc[(temp.Year==2013)&(temp.Quarter==temp.Quarter.iloc[k]),"Sales"].median() 
            else:
                temp["Prev_YrQuarter_Mdn"].iloc[k] = temp.loc[(temp.Year==2014),"Sales"].median() 
                
            if len(temp.loc[(temp.Year==2014)&(temp.WeekOfYear==temp.WeekOfYear.iloc[k])]) > 0:
                temp["Prev_YrWeek_Mdn"].iloc[k] = temp.loc[(temp.Year==2014)&(temp.WeekOfYear==temp.WeekOfYear.iloc[k]),"Sales"].median()
            elif len(temp.loc[(temp.Year==2013)&(temp.WeekOfYear==temp.WeekOfYear.iloc[k])]) > 0:
                temp["Prev_YrWeek_Mdn"].iloc[k] = temp.loc[(temp.Year==2013)&(temp.WeekOfYear==temp.WeekOfYear.iloc[k]),"Sales"].median() 
            else:
                temp["Prev_YrWeek_Mdn"].iloc[k] = temp.loc[(temp.Year==2014),"Sales"].median() 
                
        elif temp["Year"].iloc[k] == 2013:
            temp["Prev_YrMonth_Mdn"].iloc[k] = temp.loc[(temp.Year==2013),"Sales"].median()
            temp["Prev_YrQuarter_Mdn"].iloc[k] = temp.loc[(temp.Year==2013),"Sales"].median()
            temp["Prev_YrWeek_Mdn"].iloc[k] = temp.loc[(temp.Year==2013),"Sales"].median()
            
    
    temp["Promo2_Effect"] = 0
    temp["Promo2_Type"] = 0
    
    if temp.Promo2.iloc[0] == 1:
        for mnth in temp.PromoInterval.iloc[0].split(","):
            for k,v in temp.iterrows():
                if ((temp.Month.iloc[k] == m[mnth][0])&(temp.Day.iloc[k]>15)&(temp["Open"].iloc[k]==1))|((temp.Month.iloc[k] == m[mnth][1])&(temp.Day.iloc[k]<15)&(temp["Open"].iloc[k]==1)):
                    temp["Promo2_Effect"].iloc[k] = 1
        temp.loc[(temp.Year<temp.Promo2SinceYear.iloc[0])|((temp.Year==temp.Promo2SinceYear.iloc[0])&(temp.WeekOfYear<temp.Promo2SinceWeek.iloc[0])),"Promo2_Effect"] = 0
        if temp.PromoInterval.iloc[0] == "Jan,Apr,Jul,Oct":
            temp["Promo2_Type"] = 2
        elif temp.PromoInterval.iloc[0] == "Feb,May,Aug,Nov":
            temp["Promo2_Type"] = 1
        elif temp.PromoInterval.iloc[0] == "Mar,Jun,Sept,Dec":
            temp["Promo2_Type"] = 3
        
        
        df = pd.concat([df,temp])
    else:
        df = pd.concat([df,temp])

train_data = df.copy()
train_data = train_data.reset_index(drop=True)

#%%

# Avg Sales Per Customer
df_avg_sales = train_data.copy()
df_avg_sales = df_avg_sales[df_avg_sales.Customers>0]
df_avg_sales["Avg_Sales_Per_Customer"] = df_avg_sales["Sales"]/df_avg_sales["Customers"]
df_avg_sales = df_avg_sales.groupby(by=["Store"],as_index=False).agg({"Avg_Sales_Per_Customer":"mean"})

# Avg Customers while Promo
df_avg_cust_promo = train_data.copy()
df_avg_cust_promo = df_avg_cust_promo.loc[df_avg_cust_promo.Promo==1]
df_avg_cust_promo = df_avg_cust_promo.groupby(by=["Store"],as_index=False).agg({"Customers":"mean"})
df_avg_cust_promo = df_avg_cust_promo.rename(columns={"Customers":"Avg_Customers_While_Promo"})

# Avg Customers while School Holiday
df_avg_cust_scholiday = train_data.copy()
df_avg_cust_scholiday = df_avg_cust_scholiday.loc[df_avg_cust_scholiday.SchoolHoliday==1]
df_avg_cust_scholiday = df_avg_cust_scholiday.groupby(by=["Store"],as_index=False).agg({"Customers":"mean"})
df_avg_cust_scholiday = df_avg_cust_scholiday.rename(columns={"Customers":"Avg_Customers_While_Holiday"})

# Avg Days Open Per Day
df_avg_open = train_data.copy()
df_avg_open = df_avg_open.groupby(by=["Store"],as_index=False).apply(lambda x:len(x.loc[x.Open==1])/len(x))
df_avg_open = df_avg_open.rename(columns={None:"Avg_Days_Open"})

#Avg Promos Per Month
df_avg_promo_pm = train_data.copy()
df_avg_promo_pm = df_avg_promo_pm.groupby(by=["Store","Year","Month"],as_index=False).apply(lambda x:x["Promo"].sum()/len(x))
df_avg_promo_pm = df_avg_promo_pm.rename(columns={None:"Avg_Promo_Per_Month"})

#Avg School Holidays Per Month
df_avg_holiday_pm = train_data.copy()
df_avg_holiday_pm = df_avg_holiday_pm.groupby(by=["Store","Year","Month"],as_index=False).apply(lambda x:x["SchoolHoliday"].sum()/len(x))
df_avg_holiday_pm = df_avg_holiday_pm.rename(columns={None:"Avg_School_Holidays"})

#Avg School Holidays Per Week
df_avg_holiday_pw = train_data.copy()
df_avg_holiday_pw = df_avg_holiday_pw.groupby(by=["Store","Year","WeekOfYear"],as_index=False).apply(lambda x:x["SchoolHoliday"].sum()/len(x))
df_avg_holiday_pw = df_avg_holiday_pw.rename(columns={None:"Avg_School_Holidays_pw"})

# State Holidays
train_data["Stat_Holidays"] = 0
train_data.loc[(train_data.StateHoliday=='b')|(train_data.StateHoliday=='c'),"Stat_Holidays"] = 1

# Assortment, Store Type Label Encode

train_data["Assortment"] = le.fit_transform(train_data["Assortment"])
train_data["StoreType"] = le.fit_transform(train_data["StoreType"])

train_data = train_data.drop(["CO_Date","Promo2SinceWeek","Promo2SinceYear","PromoInterval",
                              "Promo2","CompetitionOpenSinceYear","CompetitionOpenSinceMonth",
                              "StateHoliday","Customers"],axis=1)

train_data = train_data.merge(df_avg_sales,how="left",on=["Store"])
train_data = train_data.merge(df_avg_cust_promo,how="left",on=["Store"])
train_data = train_data.merge(df_avg_cust_scholiday,how="left",on=["Store"])
train_data = train_data.merge(df_avg_open,how="left",on=["Store"])
train_data = train_data.merge(df_avg_promo_pm,how="left",on=["Store","Year","Month"])
train_data = train_data.merge(df_avg_holiday_pm,how="left",on=["Store","Year","Month"])
train_data = train_data.merge(df_avg_holiday_pw,how="left",on=["Store","Year","WeekOfYear"])

train_data = train_data.sort_values(by=["Avg_Sales_Per_Customer","Store","Date"])
train_data = train_data.reset_index(drop=True)

#%%
# Future Data Prep

store_data = pd.read_csv(path+"store.csv")

future_data["Year"] = future_data["Date"].dt.year
future_data["Month"] = future_data["Date"].dt.month
future_data["WeekOfYear"] = future_data["Date"].dt.week
future_data["Day"] = future_data["Date"].dt.day
future_data["DayOfYear"] = future_data["Date"].dt.dayofyear
future_data["Quarter"] = future_data["Date"].dt.quarter

# State Holidays
future_data["Stat_Holidays"] = 0
future_data.loc[(future_data.StateHoliday=='b')|(future_data.StateHoliday=='c'),"Stat_Holidays"] = 1

future_data = future_data.drop(["StateHoliday","Id"],axis=1)

last_n_train = train_data.loc[(train_data.Open==1)&(train_data.Store.isin(list(future_data.Store.unique())))]
last_n_train = last_n_train.groupby(by=["Store"],as_index=False).tail(11)
future_data = pd.concat([future_data,last_n_train])

df = pd.DataFrame()
for st in list(future_data.Store.unique()):
    temp = future_data[future_data.Store==st]
    temp = temp.sort_values(by=["Date"])
    temp = temp.reset_index(drop=True)
    
    temp["Open"] = temp["Open"].fillna(temp["Open"].mode)  
    
    temp2 = store_data[store_data.Store==st]
    temp2 = temp2.reset_index(drop=True)
    
    temp3 = train_data[train_data.Store==st]
    temp3 = temp3.sort_values(by=["Date"])
    temp3 = temp3.reset_index(drop=True)
    
    temp["Promo2_Effect"] = 0
    temp["Promo2_Type"] = 0
    
    if temp2.Promo2.iloc[0] == 1:
        for mnth in temp2.PromoInterval.iloc[0].split(","):
            for k,v in temp.iterrows():
                if ((temp.Month.iloc[k] == m[mnth][0])&(temp.Day.iloc[k]>15)&(temp["Open"].iloc[k]==1))|((temp.Month.iloc[k] == m[mnth][1])&(temp.Day.iloc[k]<15)&(temp["Open"].iloc[k]==1)):
                    temp["Promo2_Effect"].iloc[k] = 1
                    
        if temp2.PromoInterval.iloc[0] == "Jan,Apr,Jul,Oct":
            temp["Promo2_Type"] = 2
        elif temp2.PromoInterval.iloc[0] == "Feb,May,Aug,Nov":
            temp["Promo2_Type"] = 1
        elif temp2.PromoInterval.iloc[0] == "Mar,Jun,Sept,Dec":
            temp["Promo2_Type"] = 3
        

    temp = temp.drop(["Sales"],axis=1)
    temp["CompetitionDistance"] = temp["CompetitionDistance"].ffill()
    temp["Competition_Effect"] = temp["Competition_Effect"].ffill()
    temp["Affected"] = temp["Affected"].ffill()
    temp["Avg_Sales_Per_Customer"] = temp["Avg_Sales_Per_Customer"].ffill()
    temp["Avg_Customers_While_Promo"] = temp["Avg_Customers_While_Promo"].ffill()
    temp["Avg_Customers_While_Holiday"] = temp["Avg_Customers_While_Holiday"].ffill()
    temp["Avg_Days_Open"] = temp["Avg_Days_Open"].ffill()
    temp["Assortment"] = temp["Assortment"].ffill()
    temp["StoreType"] = temp["StoreType"].ffill()
    temp["t_line"] = temp["t_line"].ffill()
    
    #Avg Promos Per Month
    temp_avg_promo_pm = temp.iloc[11:].copy()
    temp_avg_promo_pm = temp_avg_promo_pm.reset_index(drop=True)
    temp_avg_promo_pm = temp_avg_promo_pm.groupby(by=["Year","Month"],as_index=False).apply(lambda x:x["Promo"].sum()/len(x)).rename(columns={None:"Avg_Promo_Per_Month"})
    temp.loc[(temp.Year==2015)&(temp.Month==8),"Avg_Promo_Per_Month"] = temp_avg_promo_pm.loc[(temp_avg_promo_pm.Year==2015)&(temp_avg_promo_pm.Month==8),"Avg_Promo_Per_Month"].iloc[0]
    temp.loc[(temp.Year==2015)&(temp.Month==9),"Avg_Promo_Per_Month"] = temp_avg_promo_pm.loc[(temp_avg_promo_pm.Year==2015)&(temp_avg_promo_pm.Month==9),"Avg_Promo_Per_Month"].iloc[0]
    
    #Avg SchoolHolidays Per Month
    temp_avg_sch_pm = temp.iloc[11:].copy()
    temp_avg_sch_pm = temp_avg_sch_pm.reset_index(drop=True)
    temp_avg_sch_pm = temp_avg_sch_pm.groupby(by=["Year","Month"],as_index=False).apply(lambda x:x["SchoolHoliday"].sum()/len(x)).rename(columns={None:"Avg_School_Holidays"})
    temp.loc[(temp.Year==2015)&(temp.Month==8),"Avg_School_Holidays"] = temp_avg_sch_pm.loc[(temp_avg_sch_pm.Year==2015)&(temp_avg_sch_pm.Month==8),"Avg_School_Holidays"].iloc[0]
    temp.loc[(temp.Year==2015)&(temp.Month==9),"Avg_School_Holidays"] = temp_avg_sch_pm.loc[(temp_avg_sch_pm.Year==2015)&(temp_avg_sch_pm.Month==9),"Avg_School_Holidays"].iloc[0]
    
    try:
        temp.loc[(temp.Year==2015)&(temp.Month==9),"Avg_Promo_Per_Month"] = temp3.loc[((temp3.Year==2014)&(temp3.Month==9)),"Avg_Promo_Per_Month"].iloc[0]
    except:
        temp.loc[(temp.Year==2015)&(temp.Month==9),"Avg_Promo_Per_Month"] = temp3.loc[((temp3.Year==2013)&(temp3.Month==9)),"Avg_Promo_Per_Month"].iloc[0]
    
    #Avg SchoolHolidays Per Week
    temp_avg_sch_wk = temp.iloc[11:].copy()
    temp_avg_sch_wk = temp_avg_sch_wk.reset_index(drop=True)
    temp_avg_sch_wk = temp_avg_sch_wk.groupby(by=["Year","WeekOfYear"],as_index=False).apply(lambda x:x["SchoolHoliday"].sum()/len(x)).rename(columns={None:"Avg_School_Holidays_pw"})
    
    for wk in list(temp_avg_sch_wk.WeekOfYear.unique()):
        temp.loc[(temp.Year==2015)&(temp.WeekOfYear==wk),"Avg_School_Holidays_pw"] = temp_avg_sch_wk.loc[(temp_avg_sch_wk.Year==2015)&(temp_avg_sch_wk.WeekOfYear==wk),"Avg_School_Holidays_pw"].iloc[0]
    
    # Median of Previous Quarter
    for yr in list(temp.Year.unique()):
        temp_yr = temp[temp.Year==yr]
        for qt in list(temp_yr.Quarter.unique()):
            if (yr==2013) & (qt==1):
                continue
            else:
                if qt == 1:
                    yr_prev = yr - 1
                    qt_prev = 4
                else:
                    yr_prev = yr
                    qt_prev = qt - 1
                temp.loc[(temp.Year==yr)&(temp.Quarter==qt),"Prev_Quarter_Mdn"] = temp3.loc[(temp3.Year==yr_prev)&
                                                                                           (temp3.Quarter==qt_prev)&
                                                                                           (temp3.Promo == 1),"Sales"].median() 
    # School Holidays Last Week and Same Month,Quarter and Week of Last Year
    for k,v in temp.iterrows():
        
        if k > 10:
            
            if len(temp3.loc[(temp3.Year==2014)&(temp3.Month==temp.Month.iloc[k])]) > 0:
                temp["Prev_YrMonth_Mdn"].iloc[k] = temp3.loc[(temp3.Year==2014)&(temp3.Month==temp.Month.iloc[k]),"Sales"].median()
            elif len(temp3.loc[(temp3.Year==2013)&(temp3.Month==temp.Month.iloc[k])]) > 0:
                temp["Prev_YrMonth_Mdn"].iloc[k] = temp3.loc[(temp3.Year==2013)&(temp3.Month==temp.Month.iloc[k]),"Sales"].median() 
            else:
                temp["Prev_YrMonth_Mdn"].iloc[k] = temp3.loc[(temp3.Year==2014),"Sales"].median()   
                
            if len(temp3.loc[(temp3.Year==2014)&(temp3.Quarter==temp.Quarter.iloc[k])]) > 0:
                temp["Prev_YrQuarter_Mdn"].iloc[k] = temp3.loc[(temp3.Year==2014)&(temp3.Quarter==temp.Quarter.iloc[k]),"Sales"].median()
            elif len(temp3.loc[(temp3.Year==2013)&(temp3.Quarter==temp.Quarter.iloc[k])]) > 0:
                temp["Prev_YrQuarter_Mdn"].iloc[k] = temp3.loc[(temp3.Year==2013)&(temp3.Quarter==temp.Quarter.iloc[k]),"Sales"].median() 
            else:
                temp["Prev_YrQuarter_Mdn"].iloc[k] = temp3.loc[(temp3.Year==2014),"Sales"].median() 
                
            if len(temp3.loc[(temp3.Year==2014)&(temp3.WeekOfYear==temp.WeekOfYear.iloc[k])]) > 0:
                temp["Prev_YrWeek_Mdn"].iloc[k] = temp3.loc[(temp3.Year==2014)&(temp3.WeekOfYear==temp.WeekOfYear.iloc[k]),"Sales"].median()
            elif len(temp3.loc[(temp3.Year==2013)&(temp3.WeekOfYear==temp.WeekOfYear.iloc[k])]) > 0:
                temp["Prev_YrWeek_Mdn"].iloc[k] = temp3.loc[(temp3.Year==2013)&(temp3.WeekOfYear==temp.WeekOfYear.iloc[k]),"Sales"].median() 
            else:
                temp["Prev_YrWeek_Mdn"].iloc[k] = temp3.loc[(temp3.Year==2014),"Sales"].median() 
                    
            
            
            temp["Sch_Holidays_Last_Wk"].iloc[k] = temp["SchoolHoliday"].iloc[k-7:k].mean() 
    
    
    df = pd.concat([df,temp])

future_data = df.copy()
future_data = future_data.sort_values(by=["Avg_Sales_Per_Customer","Store","Date"])

#%%
# Saving Prepared Train and Future Data 

train_data = train_data.loc[train_data.Open==1]
train_data.to_csv(path+"train_data.csv",index=False)

future_data = future_data.loc[future_data.Open==1]

t_columns = list(train_data.columns)
t_columns.remove("Sales")

future_data = future_data[t_columns]
future_data.to_csv(path+"future_data.csv",index=False)

