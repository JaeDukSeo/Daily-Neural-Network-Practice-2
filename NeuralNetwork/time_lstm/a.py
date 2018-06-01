import pandas as pd
import numpy as np,sys
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from numpy import newaxis

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
Mean_list_scale = scaler.fit_transform( np.expand_dims(np.asarray(Mean_list),1))
train = Mean_list_scale[:(697*10),0]
test  = Mean_list_scale[(697*10):,0]
print(Mean_list_scale.shape)

# Def: Concvert Data into the shape of the good
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(trainX.shape,trainY.shape,testX.shape,testY.shape)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(input_shape=(None,1),units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1 ,kernel_initializer ='uniform'))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer='adam')

model.fit(
    trainX,trainY,
    batch_size=10,epochs=10,
    validation_split=0.05)

def plot_results_multiple(predicted_data, true_data,length):
    plt.plot(scaler.inverse_transform(true_data.reshape(-1, 1))[length:])
    plt.plot(scaler.inverse_transform(np.array(predicted_data).reshape(-1, 1))[length:])
    plt.show()
    
#predict lenght consecutive values from a real one
def predict_sequences_multiple(model,firstValue,length):
    prediction_seqs = []
    curr_frame = firstValue
    for i in range(length): 
        predicted = []                
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[0:]
        curr_frame = np.insert(curr_frame[0:], i+1, predicted[-1], axis=0)
        prediction_seqs.append(predicted[-1])
    return prediction_seqs

predict_length=len(testX)
predictions = predict_sequences_multiple(model, testX[0], predict_length)


plt.figure()
plt.plot(df.Low,color='g',label='Low')
plt.plot(df.Mean,color='r',label='Mean')
plt.plot(df.High,color='y',label='High')
plt.plot(predictions,color='orange',label='Prediction')
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend()
plt.show()


# -- end code --