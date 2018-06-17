
# boilerplate# boile 
import codecs
import functools
import os
import tempfile
import zipfile

from nltk.tokenize import sexpr
import numpy as np
from six.moves import urllib
import tensorflow as tf
sess = tf.InteractiveSession()
# import tensorflow_fold as td


data_dir   = tempfile.mkdtemp()
print('saving files to %s' % data_dir)


def download_and_unzip(url_base, zip_name, *file_names):
    zip_path = os.path.join(data_dir, zip_name)
    url = url_base + zip_name
    print('downloading %s to %s' % (url, zip_path))
    urllib.request.urlretrieve(url, zip_path)
    out_paths = []
    with zipfile.ZipFile(zip_path, 'r') as f:
        for file_name in file_names:
            print('extracting %s' % file_name)
            out_paths.append(f.extract(file_name, path=data_dir))
    return out_paths

full_glove_path, = download_and_unzip('http://nlp.stanford.edu/data/', 'glove.840B.300d.zip','glove.840B.300d.txt')


train_pathtrain_p , dev_path, test_path = download_and_unzip(
  'http://nlp.stanford.edu/sentiment/', 'trainDevTestTrees_PTB.zip', 
  'trees/train.txt', 'trees/dev.txt', 'trees/test.txt')

filtered_glove_path = os.path.join(data_dir, 'filtered_glove.txt')

def filter_glove():
    vocab = set()
    # Download the full set of unlabeled sentences separated by '|'.
    sentence_path, = download_and_unzip('http://nlp.stanford.edu/~socherr/', 'stanfordSentimentTreebank.zip', 'stanfordSentimentTreebank/SOStr.txt')
    with codecs.open(sentence_path, encoding='utf-8') as f:
        for line in f:
            # Drop the trailing newline and strip backslashes. Split into words.
            vocab.update(line.strip().replace('\\', '').split('|'))
    nread = 0
    nwrote = 0
    with codecs.open(full_glove_path, encoding='utf-8') as f:
        with codecs.open(filtered_glove_path, 'w', encoding='utf-8') as out:
            for line in f:
                nread += 1
                line = line.strip()
                if not line: continue
                if line.split(u' ', 1)[0] in vocab:
                    out.write(line + '\n')
                    nwrote += 1
    print('read %s lines, wrote %s' % (nread, nwrote))

filter_glove()

def  load_embeddings (embedding_path):
    print('loading word embeddings from %s' % embedding_path)
    weight_vectors = []
    word_idx = {}
    with codecs.open(embedding_path, encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(u' ', 1)
            word_idx[word] = len(weight_vectors)
            weight_vectors.append(np.array(vec.split(), dtype=np.float32))
    # Annoying implementation detail; '(' and ')' are replaced by '-LRB-' and
    # '-RRB-' respectively in the parse-trees.
    word_idx[u'-LRB-'] = word_idx.pop(u'(')
    word_idx[u'-RRB-'] = word_idx.pop(u')')
    # Random embedding vector for unknown words.
    weight_vectors.append(np.random.uniform(-0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
    return np.stack(weight_vectors), word_idx

weight_matrix, word_idx = load_embeddings(filtered_glove_path)

def load_trees(filename):
    with codecs.open(filename, encoding='utf-8') as f:
        # Drop the trailing newline and strip \s.
        trees = [line.strip().replace('\\', '') for line in f]
        print('loaded %s trees from %s' % (len(trees), filename))
        return trees

train_trees = load_trees(train_path)
dev_trees = load_trees(dev_path)
test_trees = load_trees(test_path)

# -- end code --