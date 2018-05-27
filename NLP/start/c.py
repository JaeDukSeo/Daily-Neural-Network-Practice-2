import nltk 
import numpy as np
from nltk.tokenize import word_tokenize

# question 7
sentence = "The dog was big but it was so doggy and dog"
word_list = word_tokenize(sentence)
word_set = set(word_list)
freq = {}
for word in word_set:
    freq[word] = word_list.count(word) 
    print(word,' : ',freq[word])
print('------------------------------')
tagged = nltk.pos_tag(word_list)
print(tagged)
print('------------------------------')

# -- end code --