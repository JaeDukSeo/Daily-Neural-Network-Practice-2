import nltk,sys
import numpy as np,pandas as pd
import matplotlib.pyplot as plt 
import nltk.data
from nltk.stem.porter import *
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

# print(short_data['SentimentText'])
# print('-------Remove Stop Word--------')
short_data['Step1_SentimentText'] = short_data['SentimentText'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
# print(short_data['Step1_SentimentText'])



# step 2 replace special char and replace abbreviations
print('\n--------------------------------')
import csv,re

# Code From: https://medium.com/nerd-stuff/python-script-to-turn-text-message-abbreviations-into-actual-phrases-d5db6f489222
def translator(user_string):
    user_string = user_string.split(" ")
    j = 0
    for _str in user_string:
        # File path which consists of Abbreviations.
        fileName = "/Users/JDSeo/Desktop/Daily-Neural-Network-Practice-2/NLP/cleaning/slang.txt"

        # File Access mode [Read Mode]
        with open(fileName, "r") as myCSVfile:
            # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
            dataFromFile = csv.reader(myCSVfile, delimiter="=")
            # Removing Special Characters.
            _str = re.sub('[^a-zA-Z0-9]+', '', _str)
            for row in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
                if _str.upper() == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    user_string[j] = row[1]
            myCSVfile.close()
        j = j + 1
    return ' '.join(user_string)

# print(short_data['Step1_SentimentText'])
# print('-------Replace Abbreviations--------')
short_data['Step2_SentimentText'] = short_data['Step1_SentimentText'].apply(lambda x:  translator(x)  ) 
# print(short_data['Step2_SentimentText'])



# step 3 stemming 
ps = PorterStemmer()
# print(short_data['Step2_SentimentText'])
# print('-------Stemming--------')
short_data['Step3_SentimentText'] = short_data['Step2_SentimentText'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() ]))
# print(short_data['Step3_SentimentText'])





# step 4 Lemmazation
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
# print(short_data['Step2_SentimentText'])
# print('-------Lemmazation--------')
short_data['Step4_SentimentText'] = short_data['Step2_SentimentText'].apply(lambda x: ' '.join([lmtzr.lemmatize(word,'v') for word in x.split() ]))
# print(short_data['Step4_SentimentText'])




# step 5 Lemmazation
# print(short_data['Step2_SentimentText'])
# print('-------Part of Speech Tagging--------')
short_data['Step5_SentimentText'] = short_data['Step2_SentimentText'].apply(lambda x: nltk.pos_tag(nltk.word_tokenize(x)))
# print(short_data['Step5_SentimentText'])



# step 6 Capitalization
print(short_data['Step2_SentimentText'])
print('-------Capitalization--------')
short_data['Step6_SentimentText'] = short_data['Step2_SentimentText'].apply(  lambda x: ' '.join( [ word.upper() for word in x.split() ] ) )
print(short_data['Step6_SentimentText'])

# - end code -