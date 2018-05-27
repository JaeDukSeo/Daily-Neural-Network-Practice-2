import nltk 
import numpy as np
from nltk.tokenize import word_tokenize

# question 7
# sentence = "The dog was big but it was so doggy and dog"
# word_list = word_tokenize(sentence)
# word_set = set(word_list)
# freq = {}
# for word in word_set:
#     freq[word] = word_list.count(word) 
#     print(word,' : ',freq[word])
# print('------------------------------')
# tagged = nltk.pos_tag(word_list)
# print(tagged)
# print('------------------------------')



# question 10
question = "I am planning to visit New Delhi to \
attend Analytics Vidhya Delhi Hackathon"
word_list = word_tokenize(question)
word_set = set(word_list)
freq = {}
print('------------------------------')
tagged = nltk.pos_tag(word_list)
for x,y in tagged:
    if "N" in y or "PRP" in y:
        print(x)
print('------------------------------')
for x,y in tagged:
    if "V" in y:
        print(x)
print('------------------------------')
for word in word_set:
    freq[word] = word_list.count(word) 
    if freq[word] > 1:
        print(word,' : ',freq[word])
# -- end code --