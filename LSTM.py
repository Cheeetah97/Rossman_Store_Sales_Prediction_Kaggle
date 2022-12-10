import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

#%%
# Loading Train,Test and Future arrays along with Scaler Objects

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
Y_train = np.load("Y_train.npy")
Y_test = np.load("Y_test.npy")
X_fut = np.load("X_fut.npy")

orig_train_df = pd.read_csv("orig_train_df.csv")
orig_test_df = pd.read_csv("orig_test_df.csv")
orig_future_df = pd.read_csv("orig_future_df.csv")
stores = pd.read_csv("Store_Labels.csv",)

orig_train_df["Date"] = pd.to_datetime(orig_train_df["Date"],format="%Y-%m-%d")
orig_test_df["Date"] = pd.to_datetime(orig_test_df["Date"],format="%Y-%m-%d")
orig_future_df["Date"] = pd.to_datetime(orig_future_df["Date"],format="%Y-%m-%d")

x_scaler = joblib.load('xscaler.save') 
y_scaler = joblib.load('yscaler.save')

#%%
# Stateless LSTM Architecture
# Look back window 12 Days
# Layers 3:- LSTM(125)-->LSTM(75)-->LSTM(50)-->Dense(12)-->Dense(1)
# Early Stopping Rounds 50
# Batch Size 900

activation_function = "tanh"
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = 'msle'
num_epochs = 500

# Initialize the RNN
model = tf.keras.models.Sequential()
    
# Adding the input layer and the LSTM layers
model.add(tf.keras.layers.LSTM(units=125,activation=activation_function,return_sequences=True,input_shape=(12,25)))
#model.add(tf.keras.layers.Dropout(0.20))
model.add(tf.keras.layers.LSTM(units=75,activation=activation_function,return_sequences=True))
#model.add(tf.keras.layers.Dropout(0.20))
model.add(tf.keras.layers.LSTM(units=50,activation=activation_function))
#model.add(tf.keras.layers.Dropout(0.20))
model.add(tf.keras.layers.Dense(units =12,activation=activation_function))
#model.add(tf.keras.layers.Dropout(0.20))
model.add(tf.keras.layers.Dense(units = 1,activation='relu'))

# Compiling the RNN
model.compile(optimizer=optimizer,loss=loss_function,metrics=['mse'])

# Early stopping and mode save
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = tf.keras.callbacks.ModelCheckpoint('trained_model.h5',monitor='val_loss',mode='min',verbose=1,save_best_only=True)

model.fit(X_train,Y_train,validation_data=(X_test,Y_test),batch_size=900,epochs=num_epochs,callbacks=[es,mc])

#%%
# Generating Predictions

tf.keras.backend.clear_session()
saved_model = tf.keras.models.load_model('trained_model.h5')

test_pred = saved_model.predict(X_test)
test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape((len(test_pred),1)))

future_pred = saved_model.predict(X_fut)
future_pred = y_scaler.inverse_transform(np.array(future_pred).reshape((len(future_pred),1)))

orig_test_df["Predicted"] = test_pred
orig_test_df["Portion"] = "test"

orig_future_df["Predicted"] = future_pred
orig_future_df["Sales"] = np.nan
orig_future_df["Portion"] = "future"

final_df = pd.concat([orig_test_df,orig_future_df])
final_df = final_df.rename(columns={"Store":"Code"})
final_df = final_df.merge(stores,how="left",on=["Code"])
final_df = final_df.drop("Code",axis=1)
final_df.to_csv("Final_Results_3_msle.csv",index=False)

to_upload = pd.read_csv("test.csv")
to_upload["Date"] = pd.to_datetime(to_upload["Date"],format="%Y-%m-%d")
to_upload = to_upload.merge(final_df[["Date","Store","Predicted"]],how="left",on=["Date","Store"])
to_upload["Predicted"] = to_upload["Predicted"].fillna(0)
to_upload = to_upload.rename(columns={"Predicted":"Sales"})
to_upload[["Id","Sales"]].to_csv("Upload_3_msle.csv",index=False)
