import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load subset of the training data
path = '../../Dataset/TalkingDataAdTracking/'
X_train = pd.read_csv(path+'train.csv', nrows=1000000, parse_dates=['click_time'])

# Show the head of the table
print(X_train.head())
print('---------------------------------------')

# Extract the data hour minute etc - conversion
X_train['day'] = X_train['click_time'].dt.day.astype('uint8')
X_train['hour'] = X_train['click_time'].dt.hour.astype('uint8')
X_train['minute'] = X_train['click_time'].dt.minute.astype('uint8')
X_train['second'] = X_train['click_time'].dt.second.astype('uint8')
print(X_train.head())
print('---------------------------------------')




# -- end code --