# generate price correlation matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# load package
from cryptory import Cryptory
import datetime

# initialise object 
# pull data from start of 2017 to present day
my_cryptory = Cryptory(from_date = "2016-01-01")

# get historical bitcoin prices from coinmarketcap
my_cryptory.extract_coinmarketcap("bitcoin")
all_coins_df = my_cryptory.extract_bitinfocharts("btc")
# coins of interest
bitinfocoins = ["btc", "eth", "xrp", "bch", "ltc", "dash", "xmr", "doge"]
for coin in bitinfocoins[1:]:
    all_coins_df = all_coins_df.merge(my_cryptory.extract_bitinfocharts(coin), on="date", how="left")
# date column not need for upcoming calculations
all_coins_df = all_coins_df.drop('date', axis=1)
corr = all_coins_df.pct_change().corr(method='pearson')
# fig, ax = plt.subplots(figsize=(7,5))  
# sns.heatmap(corr, 
#             xticklabels=[col.replace("_price", "") for col in corr.columns.values],
#             yticklabels=[col.replace("_price", "") for col in corr.columns.values],
#             annot_kws={"size": 16})
# plt.show()

# overlay bitcoin price and google searches for bitcoin
btc_google = my_cryptory.get_google_trends(kw_list=['bitcoin']).merge(
    my_cryptory.extract_coinmarketcap('bitcoin')[['date','close']], 
    on='date', how='inner')

# need to scale columns (min-max scaling)
btc_google[['bitcoin','close']] = (
        btc_google[['bitcoin', 'close']]-btc_google[['bitcoin', 'close']].min())/(
        btc_google[['bitcoin', 'close']].max()-btc_google[['bitcoin', 'close']].min())

fig, ax1 = plt.subplots(1, 1, figsize=(9, 3))
ax1.set_xticks([datetime.date(j,i,1) for i in range(1,13,2) for j in range(2017,2019)])
ax1.set_xticklabels([datetime.date(j,i,1).strftime('%b %d %Y')  for i in range(1,13,2) for j in range(2017,2019)])
ax1.plot(btc_google['date'].astype(datetime.datetime),
             btc_google['close'], label='bitcoin', color='#FF9900')
ax1.plot(btc_google['date'].astype(datetime.datetime),
             btc_google['bitcoin'], label="bitcoin (google search)", color='#4885ed')
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., ncol=2, prop={'size': 14})
plt.show()