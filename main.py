import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import SGD
from datetime import date
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import streamlit as st
plt.rcParams["figure.figsize"] = (20,10)
# st.set_option('deprecation.showPyplotGlobalUse', False)

# import nsepy
# from nsepy import get_history
# from datetime import date
# data = get_history(symbol="IOC", start=date(2017,1,1), end=date(2019,2,15))
# data[['Close']].plot()

st.markdown(''' 
# Assistant Analyst :sunglasses: <br>
''',True)   #true is passesd so that <breakline> works

stock = st.sidebar.selectbox(
    'Select the stock : ',
     ['INFY','ASIANPAINT','BAJFINANCE','HDFC','TCS','TATAMOTORS']
)

start_date = st.sidebar.date_input('Enter starting point of data')
end_date = date.today()
today = date.today()
lookback = st.sidebar.slider('Lookback : ',min_value = 1,max_value = 100,step = 1)
usdXinr = web.DataReader("INR=X", 'yahoo').iloc[-1]['Close']


df = web.DataReader(stock,data_source = "yahoo" , start = start_date,end = end_date)

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

train_size = (int)(0.8*len(dataset))
train , test = dataset[:train_size,:] , dataset[train_size - lookback:,:]

def to_sequences(dataset,lookback = 15):
    x = []
    y = []
    for i in range(0,len(dataset) - lookback - 1) :
        window = dataset[i:i + lookback,0]
        x.append(window)
        y.append(dataset[i + lookback,:])
    return np.array(x) , np.array(y)

trainx,trainy = to_sequences(train,lookback)
testx,testy = to_sequences(test,lookback)
print(trainy.shape)

# building the model

model = Sequential()
model.add(Dense(32, input_dim = lookback , activation = 'relu'))
model.add(Dense(16 , activation = 'relu'))

model.add(Dense(1))
# trainy[0].shape = number of days for which we will predict , currently 1

#opt = SGD(learning_rate = 0.1, momentum = 0.9)

model.compile(loss = 'mean_squared_error' , optimizer = 'adam' , metrics = ['acc'])  # try stochastic decend also

print(model.summary())


history = model.fit(trainx,trainy,validation_data = (testx,testy) , verbose = 1,epochs = 50)

#plotting accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss) + 1)
plt.plot(epochs,loss,'yellow',label = 'Training loss')
plt.plot(epochs,val_loss,'red',label = 'Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



trainpredict = scaler.inverse_transform(model.predict(trainx))
testpredict = scaler.inverse_transform(model.predict(testx))
#print(trainpredict)
trueytrain = scaler.inverse_transform(trainy)
trueytest = scaler.inverse_transform(testy)
dataset = scaler.inverse_transform(dataset)

trainScore = math.sqrt(mean_squared_error(trueytrain , trainpredict))
testScore = math.sqrt(mean_squared_error(trueytest , testpredict))
#print(trueytest[:5] ,'\n\n', testpredict[:5])
#print(len(trueytest) ,' ', len(testpredict))
print('Train Score: ',trainScore * usdXinr)
print('Test Score: ',testScore * usdXinr)

# plotting the data
trainpredplot = np.empty_like(dataset)
trainpredplot[:,:] = np.nan
trainpredplot[lookback : lookback + len(trainpredict),:] = trainpredict

testpredplot = np.empty_like(dataset)
testpredplot[:,:] = np.nan
testpredplot[len(trainx) + lookback : len(trainx) + lookback + len(testpredict),:] = testpredict

plt.title('Final Plots')
plt.xlabel('Day Number')
plt.ylabel('Price in $')
plt.plot(dataset, linewidth = 2)
plt.plot(trainpredplot , linewidth = 2)
plt.plot(testpredplot , linewidth = 2)
plt.legend(['Original', 'Prediction on training', 'Prediction on testing'])
plt.figure().figsize = (20,10)
# plt.show()
st.pyplot(plt.show())
#try scatter plot also


#predicting tomorrow's price

df = web.DataReader(stock,data_source = "yahoo" , start = start_date,end = today)[['Close']]

x = []
x.append(scaler.fit_transform(df.values)[-lookback:,0])
x = np.array(x)

trainpredict = scaler.inverse_transform(model.predict(x))

st.success('Tomorrow\'s Price :  ' + str(trainpredict[0][0]*usdXinr) + '  rupees')