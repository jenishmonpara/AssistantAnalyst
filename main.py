import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from datetime import date
import plotly.graph_objs as go
import streamlit as st
import yfinance as yf

def leave_lines(x):
    for i in range(0,x):
        st.write('.')

st.markdown(''' # Assistant Analyst :chart::money_with_wings::chart_with_downwards_trend::chart_with_upwards_trend: <br>''',True)   #true is passesd so that <breakline> works

stock = st.sidebar.selectbox(
    'Select the stock : ',
     ['ASIANPAINT.NS','BAJFINANCE.NS','CDSL.NS',
      'HCLTECH.NS','HDFC.NS','HINDUNILVR.NS','INFY.NS','ITC.NS','RELIANCE.NS',
      'TATAMOTORS.NS','TATASTEEL.NS','TITAN.NS','TCS.NS']
)

start_date = st.sidebar.date_input('Enter starting point of data',min_value = date(2000,9,9),value = date(2015,9,9))
end_date = date.today()
today = date.today()
lookback = st.sidebar.slider('Lookback : ',min_value = 1,value = 30,max_value = 100,step = 1)

# usdXinr = web.DataReader("INR=X", 'yahoo').iloc[-1]['Close']
# df = web.data.get_data_yahoo(stock,start_date,end_date)
df = yf.download(stock, start=start_date, end=end_date)


fig = go.Figure(data = [go.Candlestick(
    x = df.reset_index()['Date'],
    open = df['Open'],
    high = df['High'],
    low = df['Low'],
    close = df['Close']
    )]
)

st.write('Data : ')
st.dataframe(df, width=1200, height=500)
leave_lines(5)
st.write(stock)
fig.update_layout(height = 700)
st.plotly_chart(fig,use_container_width = True)
st.write('Volumes : ')
st.line_chart(df['Volume'])


df = df[['Close']]


dataset = df.values
dataset = dataset.astype('float64')
scaler = MinMaxScaler(feature_range = (0,1));
dataset = scaler.fit_transform(dataset)

def to_sequences(dataset,lookback = 15):
    x = []
    y = []
    for i in range(0,len(dataset) - lookback) :
        window = dataset[i:i + lookback,0]
        x.append(window)
        y.append(dataset[i + lookback,:])
    return np.array(x) , np.array(y)

trainx,trainy = to_sequences(dataset,lookback)
print(trainx.shape)
print(trainy.shape)

# building the model

model = Sequential()
model.add(Dense(32, input_dim = lookback , activation = 'relu'))
model.add(Dense(16 , activation = 'relu'))
model.add(Dense(1))
# trainy[0].shape = number of days for which we will predict , currently 1

#opt = SGD(learning_rate = 0.1, momentum = 0.9)

model.compile(loss = 'mean_squared_error' , optimizer = 'adam')

print(model.summary())


# history = model.fit(trainx,trainy,validation_data = (testx,testy) , verbose = 1,epochs = 50)
history = model.fit(trainx,trainy , verbose = 2,epochs = 50)


trainpredict = scaler.inverse_transform(model.predict(trainx))
trueytrain = scaler.inverse_transform(trainy)
dataset = scaler.inverse_transform(dataset)
trainScore = math.sqrt(mean_squared_error(trueytrain , trainpredict))


#predicting tomorrow's price

# df = web.DataReader(stock,data_source = "yahoo" , start = start_date,end = today)[['Close']]
df = yf.download(stock, start=start_date, end=today)[['Close']]

x = []
x.append(scaler.fit_transform(df.values)[-lookback:,0])
x = np.array(x)

trainpredict = scaler.inverse_transform(model.predict(x))

leave_lines(5)
st.write('Training Error (RMSE relative to stock price):  ' + str(100.0*trainScore/df.iloc[-1]['Close']) + ' %\n')
leave_lines(2)
st.success('Tomorrow\'s Price :  ' + str(trainpredict[0][0]) + '  rupees')

print(trainScore)
print(trainpredict[0][0])
