import nltk,sys
import numpy as np,pandas as pd
import matplotlib.pyplot as plt 
pd.options.mode.chained_assignment = None

# step 0 basic reading
df = pd.read_csv('train.csv',encoding='latin-1')

# print('--- Print the Basic Info of the data ----')
# print(df.info())
# print(df.shape)

# print('--- Print the Head/Tail of the data -----')
# print(df.head())
# print('------------------------')
# print(df.tail())

# df['Sentiment'].plot(kind='hist')
# plt.show()


# step 1 stop word removal
short_data = df.head()
from nltk.corpus import stopwords
stop = stopwords.words("english")

print(short_data['SentimentText'])
print('-------Remove Stop Word--------')
short_data['Step1_SentimentText'] = short_data['SentimentText'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print(short_data['Step1_SentimentText'])

# - end code -