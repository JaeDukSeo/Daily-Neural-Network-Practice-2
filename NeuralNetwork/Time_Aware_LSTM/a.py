import tensorflow as tf
import numpy as np, datetime
from matplotlib import pyplot as plt
from cryptory import Cryptory
import pandas as pd

tf.set_random_seed(678)
np.random.seed(567)

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
print(df.describe())
print(df.info())
print(df.head())
print(df.tail())


df.plot()
plt.show()

# fig, ax1 = plt.subplots(1, 1, figsize=(9, 3))
# ax1.set_xticks([datetime.date(j,i,1) for i in range(1,13,2) for j in range(2017,2019)])
# ax1.set_xticklabels([datetime.date(j,i,1).strftime('%b %d %Y') 
#                      for i in range(1,13,2) for j in range(2017,2019)])
# ax1.plot(btc_google['date'].astype(datetime.datetime),
#              btc_google['close'], label='bitcoin', color='#FF9900')
# ax1.plot(btc_google['date'].astype(datetime.datetime),
#              btc_google['bitcoin'], label="bitcoin (google search)", color='#4885ed')
# ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., ncol=2, prop={'size': 14})
# plt.show()

# -- end code --