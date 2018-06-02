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

print(df.head())



# -- end code --