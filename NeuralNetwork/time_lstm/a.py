import pandas as pd
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go

# 0. Get the Data
df = pd.read_csv('../../Dataset/Data/Stocks/aapl.us.txt',delimiter=',',usecols=['Date','Open','High','Low','Close']).sort_values('Date')

# 1. View the data
# print(df.info())
# print(df.describe())
# print(df.head())
# print(df.tail())
df.Date = pd.to_datetime(df.Date)
# df.set_index('Date',inplace=True)

# 2. create the mean of the data
df['Mean'] = (df.High + df.Low)/2.0
# df[['High','Low','Mean']].plot()
# plt.show()

# 3. Split the Data and normalize by the window size
Mean_list = df.Mean.values
a_original_data = df.Mean.values

# 4. Normalize by slide window (Normalize)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_data = np.expand_dims(Mean_list[:6273],axis=1)
test_data =  np.expand_dims(Mean_list[6273:],axis=1)

smoothing_window_size = 2091
for di in range(0,len(train_data),smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])
test_data = scaler.fit(test_data).transform(test_data)
print(train_data.shape)
print(test_data.shape)
b_norm_data =  np.concatenate([train_data,test_data],axis=0)

# 5. Moving Average to smoothenout the training data (only)
EMA = 0.0
gamma = 0.1
for ti in range(len(train_data)):
  EMA = gamma*train_data[ti,0] + (1-gamma)*EMA
  train_data[ti,0] = EMA
c_norm_data_train =  np.concatenate([train_data,test_data],axis=0)


# 6. Moving Window Average Method
window_size = 100
N = train_data.size
std_avg_predictions = list()
std_avg_x = list()
mse_errors = list()

for pred_idx in range(window_size,N):

    print(pred_idx)
    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx,'Date']

    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)
    input()

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))


# -- end code --