import pandas as pd
import numpy as np,sys
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
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = np.expand_dims(Mean_list[:6273],axis=1)
test_data =  np.expand_dims(Mean_list[6273:],axis=1)

# smoothing_window_size = 2091
# for di in range(0,len(train_data),smoothing_window_size):
#     scaler.fit(train_data[di:di+smoothing_window_size,:])
#     train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])
# test_data = scaler.fit(test_data).transform(test_data)
# b_norm_data =  np.concatenate([train_data,test_data],axis=0)

# 5. Moving Average to smoothenout the training data (only)
EMA = 0.0
gamma = 0.1
for ti in range(len(train_data)):
  EMA = gamma*train_data[ti,0] + (1-gamma)*EMA
  train_data[ti,0] = EMA
c_norm_data_train =  np.concatenate([train_data,test_data],axis=0)

# 5. Standard Average of Window
window_size = 100
N = Mean_list.size
std_avg_predictions = list(Mean_list[:window_size])
std_avg_x = list()
abs_error = list()

for pred_idx in range(window_size,N):
    std_avg_predictions.append(np.mean(Mean_list[pred_idx-window_size:pred_idx]))
    abs_error.append( np.abs(std_avg_predictions[-1]-Mean_list[pred_idx]))
print('MABSE error for standard averaging: %.5f'%(0.5*np.mean(abs_error)))

# plt.figure()
# plt.plot(Mean_list,color='b',label='True')
# plt.plot(df.Low,color='g',label='True')
# plt.plot(df.Mean,color='r',label='True')
# plt.plot(df.High,color='y',label='True')
# plt.plot(std_avg_predictions,color='orange',label='Prediction')
# plt.xlabel('Date')
# plt.ylabel('Mid Price')
# plt.legend()
# plt.show()

# 6. EXP Average of Window
window_size = 100
N = Mean_list.size

run_avg_predictions = []
run_avg_x = []

abs_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)
decay = 0.9

for pred_idx in range(1,N):
    running_mean = running_mean*decay + (1.0-decay)*Mean_list[pred_idx-1]
    run_avg_predictions.append(running_mean)
    abs_errors.append(np.abs(run_avg_predictions[-1]-Mean_list[pred_idx]))

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(abs_errors)))

# plt.figure()
# plt.plot(Mean_list,color='b',label='True')
# plt.plot(df.Low,color='g',label='True')
# plt.plot(df.Mean,color='r',label='True')
# plt.plot(df.High,color='y',label='True')
# plt.plot(run_avg_predictions,color='orange',label='Prediction')
# plt.xlabel('Date')
# plt.ylabel('Mid Price')
# plt.legend()
# plt.show()





# 7. LSTM 
print(max(Mean_list))
print(min(Mean_list))
Mean_list_scale = scaler.fit_transform( np.expand_dims(np.asarray(Mean_list),1))
print(Mean_list_scale.min())
print(Mean_list_scale.max())

plt.plot(Mean_list,color='red')
plt.show()

plt.plot(Mean_list_scale)
plt.show()

sys.exit()





# -- end code --