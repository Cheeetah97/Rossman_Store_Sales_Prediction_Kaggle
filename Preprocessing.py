import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import preprocessing

path = "M:/NN_and_Deep_Learning/Group_Project/Data/"

train_data = pd.read_csv(path+"train.csv")
test_data = pd.read_csv(path+"test.csv") 
store_data = pd.read_csv(path+"store.csv")

#%%
# Number of Stores
stores_test = test_data["Store"].unique()
stores_train = train_data["Store"].unique()

print("Total Stores in Test Data = ",test_data["Store"].nunique())
print("Total Stores in Train Data = ",train_data["Store"].nunique())
print("Stores in test but not in train = ",test_data.loc[~(test_data.Store.isin(stores_train)),"Store"].nunique())
print("Stores in train but not in test = ",train_data.loc[~(train_data.Store.isin(stores_test)),"Store"].nunique())
#%%

st_encoder = preprocessing.LabelEncoder()
ass_encoder = preprocessing.LabelEncoder()


store_data["StoreType"] = st_encoder.fit_transform(store_data["StoreType"])
store_data["Assortment"] = ass_encoder.fit_transform(store_data["Assortment"])

#plt.scatter(store_data["StoreType"],store_data["Assortment"])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(store_data["StoreType"],store_data["Assortment"],store_data["CompetitionDistance"]);