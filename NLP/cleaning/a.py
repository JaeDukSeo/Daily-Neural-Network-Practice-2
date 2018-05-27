import nltk,sys
import numpy as np,pandas as pd
import matplotlib.pyplot as plt 

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
print(short_data)
stops = set(stopwords.words("english"))
short_data['SentimentText'].apply(lambda x: [item for item in x if item not in stops])
print(short_data)

# - end code -