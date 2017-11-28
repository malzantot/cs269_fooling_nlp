"""
Author: Moustafa Alzantot (malzantot@ucla.edu)

CNN text classification model.
based on the tutorial:  http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
"""
import numpy as np
import pdb
from sklearn.datasets import fetch_20newsgroups
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import itertools
import collections

from data_utils import clean_str
#from sklearn.svm import SVC

def map_string(string, word_to_id):
    return [word_to_id[w] if w in word_to_id.keys() else word_to_id['UNK'] for w in string.split(' ')]

def remap_string(encoding, id_to_word):
    return [id_to_word[code] for code in encoding]

if __name__ == '__main__':
    newsgroups_train = fetch_20newsgroups(subset='train')
    train_targets = newsgroups_train.target
    target_names = newsgroups_train.target_names
    train_data = newsgroups_train.data

    newsgroups_test = fetch_20newsgroups(subset='test')
    test_data = newsgroups_test.data
    test_targets = newsgroups_test.target

    ## Preprocess the data
    clean_train_data = [clean_str(x) for x in train_data]
    max_seq_len = max([len(x.split(" ")) for x in clean_train_data])
    # TODO(malzantot): The line below doesnot work, it flatten it as array of chars
    # all_words = list(itertools.chain(*clean_train_data))
    all_words = []
    for doc in train_data:
        all_words.extend(doc.split(" "))
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # TOP words
    # print('Most frequent words:\n', count_pairs[0:10])
    print(count_pairs[2])
    words, cnts = list(zip(*count_pairs))
    #pdb.set_trace()

    freq_threshold = 20  # used to drop rare words
    vocab_size = len([x for x in cnts if x > freq_threshold]) + 1
    word_to_id = dict(zip(words[:vocab_size-1], range(vocab_size-1)))
    # Unknown words
    UNK = vocab_size
    PAD = vocab_size + 1
    word_to_id['UNK'] = UNK
    id_to_word = {code:w for (w, code) in word_to_id.items()}
    # print('Least frequent words:\n', count_pairs[-100:-90])
    #print('Number of words ', len(words))
    #
    train_X = [map_string(string, word_to_id) for string in clean_train_data]

    # Now, we need to do the padding !
    seq_len = [len(x) for x in train_X]
    max_seq_len = max(seq_len)
    padded_train_X = []
    for i, x in enumerate(train_X):
        padded_x = x + [PAD] * (max_seq_len - seq_len[i]) 
        padded_train_X.append(padded_x)
    

    X = np.array(padded_train_X)
    Y = np.array(train_targets)
    np.save('train_x.npy', X)
    np.save('train_y.npy', Y)
    print(X.shape)
    print(Y.shape)
    print('!! data saved !')
