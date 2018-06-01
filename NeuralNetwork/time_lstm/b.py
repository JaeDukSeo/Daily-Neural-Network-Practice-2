import pandas as pd
import numpy as np,sys
import tensorflow as tf
import matplotlib.pyplot as plt

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

import matplotlib.pyplot as plt
from numpy import newaxis

# 0. Get the Data and simple sorting and check NaN
df = pd.read_csv('../../Dataset/aapl.csv',delimiter=',',usecols=['Date','Open','High','Low','Close'])
df.Date = pd.to_datetime(df.Date)
print('Is there any Null?  : ',df.isnull().values.any())
df = df.sort_values('Date')
print(df.head())
print(df.tail())
print('------------')
print(df.info())
print('------------')
print(df.describe())

# 1. Create the Mean Value and plot
df['Mean'] = (df.High+df.Low)/2.0

trace1 = go.Scatter(
    x = df.Date,y = df.Close,
    name = 'Close',mode='line'
)
trace2 = go.Scatter(
    x = df.Date,y = df.Open,
    name = 'Open',mode='line'
)
trace3 = go.Scatter(
    x = df.Date,y = df.High,
    name = 'High',mode='line'
)
trace4 = go.Scatter(
    x = df.Date,y = df.Low,
    name = 'Low',mode='line'
)
trace5 = go.Scatter(
    x = df.Date,y = df.Mean,
    name = 'Mean',mode='line'
)
data = [trace1,trace2,trace3,trace4,trace5]
plot(data)











# -- end code --