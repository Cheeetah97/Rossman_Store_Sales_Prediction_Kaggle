import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

results = pd.read_csv("E:/Masters/NN_and_Deep_Learning/Group_Project/Data/Results/Final_Results_3_msle.csv")
results["Date"] = pd.to_datetime(results["Date"],format="%Y-%m-%d")
results = results.sort_values(by=["Store","Date"])
results = results.reset_index(drop=True)

#%%
# PostProcessing some of the Stores on after visuzalizing the predictions as the model was constantly predicting
# some stores either Low or High, in which case predictions were shifted by a fixed amount.

exception = [183]
final_df = pd.DataFrame()

for st in list(results.Store.unique()):
    temp = results[results.Store==st]
    temp = temp.reset_index(drop=True)
    
    temp["Predicted_New"] = temp["Predicted"]
    
    if st in exception:
        final_df = pd.concat([final_df,temp])
        continue
    
    if len(temp.loc[(temp.Predicted<(temp.Sales-(temp.Sales*0.20)))&(temp.Portion=="test")]) > 0.30*len(temp.loc[temp.Portion=="test"]):
        for dy in list(temp.DayOfWeek.unique()):
            temp2 = temp.loc[temp.DayOfWeek==dy]
            if len(temp2.loc[(temp2.Predicted<(temp2.Sales-(temp2.Sales*0.1)))&(temp2.Portion=="test")]) > 0:
                mean_dif = (temp2.loc[(temp2.Predicted<(temp2.Sales-(temp2.Sales*0.1)))&(temp2.Portion=="test"),"Sales"] - temp2.loc[(temp2.Predicted<(temp2.Sales-(temp2.Sales*0.1)))&(temp2.Portion=="test"),"Predicted"]).median()
                temp.loc[(temp.Portion=="future")&(temp.DayOfWeek==dy),"Predicted_New"] = temp.loc[(temp.Portion=="future")&(temp.DayOfWeek==dy),"Predicted"] + mean_dif
    
    elif len(temp.loc[(temp.Sales<(temp.Predicted-(temp.Predicted*0.20)))&(temp.Portion=="test")]) > 0.30*len(temp.loc[temp.Portion=="test"]):
        for dy in list(temp.DayOfWeek.unique()):
            temp2 = temp.loc[temp.DayOfWeek==dy]
            if len(temp2.loc[(temp2.Sales<(temp2.Predicted-(temp2.Predicted*0.1)))&(temp2.Portion=="test")]) > 0:
                mean_dif = (temp2.loc[(temp2.Sales<(temp2.Predicted-(temp2.Predicted*0.1)))&(temp2.Portion=="test"),"Predicted"] - temp2.loc[(temp2.Sales<(temp2.Predicted-(temp2.Predicted*0.1)))&(temp2.Portion=="test"),"Sales"]).median()
                temp.loc[(temp.Portion=="future")&(temp.DayOfWeek==dy),"Predicted_New"] = temp.loc[(temp.Portion=="future")&(temp.DayOfWeek==dy),"Predicted"] - mean_dif
        
    final_df = pd.concat([final_df,temp])
    
final_df.to_csv("E:/Masters/NN_and_Deep_Learning/Group_Project/Data/Results/Post_Processed_Results.csv",index=False)
    
to_upload = pd.read_csv("E:/Masters/NN_and_Deep_Learning/Group_Project/Data/test.csv")
to_upload["Date"] = pd.to_datetime(to_upload["Date"],format="%Y-%m-%d")
to_upload = to_upload.merge(final_df[["Date","Store","Predicted_New"]],how="left",on=["Date","Store"])
to_upload["Predicted_New"] = to_upload["Predicted_New"].fillna(0)
to_upload = to_upload.rename(columns={"Predicted_New":"Sales"})
to_upload[["Id","Sales"]].to_csv("E:/Masters/NN_and_Deep_Learning/Group_Project/Data/Results/Upload_Final.csv",index=False)