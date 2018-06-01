import pandas as pd
import numpy as np,sys
import tensorflow as tf
import matplotlib.pyplot as plt

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

import matplotlib.pyplot as plt
from numpy import newaxis

# 0. Get the Data
df = pd.read_csv('../../Dataset/aapl.csv',delimiter=',',usecols=['Date','Open','High','Low','Close'])
df.Date = pd.to_datetime(df.Date)
df = df.sort_values('Date')

print(df.head())
print(df.tail())
print('------------')
print(df.info())
print('------------')
print(df.describe())

data = [go.Scatter(x=df.Date,y=df.High)]
plot(data)
# -- end code --