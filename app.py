# creating a streamlit model

import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import load_model
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt


st.title('Stock Trend Prediciton')
# st.markdown()

user_input = st.text_input('Enter the stock you want to predict','AAPL')
start = "2010-01-01"
end = '2021-12-12'
df= yf.download(str(user_input),start,end)


st.subheader('Data from 2010 - 2022')
st.write(df.describe())


# vizualizations

st.subheader('Close Price Vs Time')
fig = plt.figure(figsize=(10,6))
plt.plot(df.Close)
plt.xlabel('time')
plt.ylabel('price')
st.pyplot(fig)

st.subheader('Moving Averages')
fig = plt.figure(figsize=(10,6))
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.xlabel('time')
plt.ylabel('price')
plt.legend(['100days','200days'])
st.pyplot(fig)



# we have to train model for every stock and then predict

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_scaled = scaler.fit_transform(data_training)


model = load_model('keras_model.h5')


past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)

y_predicted = model.predict(x_test)


scale_factor = 1/scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



st.subheader('Predictions')
fig2 = plt.figure(figsize=(10,6))
plt.plot(y_predicted,'r')
plt.plot(y_test,'g')
plt.legend(['predictions','real'])
plt.xlabel('time')
plt.ylabel('price')
st.pyplot(fig2)