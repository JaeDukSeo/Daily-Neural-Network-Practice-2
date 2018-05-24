from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 
import nltk
# question 1
word1 = "studies"
word2 = "studying"
stem = PorterStemmer()
lem = WordNetLemmatizer()

# print('Original Word studies (stem) : '
# ,stem.stem(word1))
# print('Original Word studies (lemm) : '
# ,lem.lemmatize(word1,"v"))

# print('Original Word studying (stem) : '
# ,stem.stem(word2))
# print('Original Word studying (lemm): '
# ,lem.lemmatize(word2,"v"))


from nltk.tokenize import word_tokenize
# quesiton 2
sentence = "Analytics Vidhya is a great source  \
to learn data science"
tokens = nltk.word_tokenize(sentence)
bigrm = list(nltk.bigrams(tokens))
count = 1
for x in bigrm:
    print(count, x)
    count = count + 1





# -- end code --