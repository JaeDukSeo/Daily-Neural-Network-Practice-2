import tensorflow as tf
import numpy as np, datetime
from matplotlib import pyplot as plt
from cryptory import Cryptory
import pandas as pd

tf.set_random_seed(678)
np.random.seed(567)


def tf_sigmoid(x): return tf.sigmoid(x)
def tf_tanh(x): return tf.nn.tanh(x)

class Time_LSTM():
    
    def __init__(self,timestamp,inc,outc):
        self.w = 

    def feed(self,input,time):
        pass
    



# data code from: https://github.com/dashee87/cryptory
# my_cryptory = Cryptory(from_date = "2017-01-01")
# btc_google = my_cryptory.get_google_trends(kw_list=['bitcoin']).merge(
#     my_cryptory.extract_coinmarketcap('bitcoin')[['date','close']], on='date', how='inner')

# # need to scale columns (min-max scaling)
# btc_google[['bitcoin','close']] = (
#         btc_google[['bitcoin', 'close']]-btc_google[['bitcoin', 'close']].min())/(
#         btc_google[['bitcoin', 'close']].max()-btc_google[['bitcoin', 'close']].min()
#     )
# btc_google = btc_google.iloc[::-1]
# btc_google.to_csv('temp.csv')

# data from csv
df = pd.read_csv('temp.csv')

# seperate the data
bitcoin_prices = df.bitcoin.values
search         = df.close.values
dates          = df.date.values




print(bitcoin_prices.shape)
print(search.shape)
print(dates.shape)


# -- end code --