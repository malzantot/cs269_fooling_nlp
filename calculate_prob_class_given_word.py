'''
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

"""
to use the probability matrix created to find p(class|word):
with open('probability_dic.p', 'rb') as handle:
    dic = pickle.dump(handle)
p_class_given_word = dic[word][class_index]
ex. dic['apple'][0]

"""

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')

word_prob = {}


with open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), 'r', encoding='UTF-8', newline='') as f:
    for line in f:
        values = line.split()
        word = values[0]
        word_prob[word] = [0]*20
#f.close()

labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids


for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        texts = []
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(texts)
        class_wc = tokenizer.word_counts
        for word, word_count in class_wc.items():
            if word in word_prob:
                word_prob[word][label_id] = word_count
for word in word_prob:
    wc_acrossclass = float(sum(word_prob[word]))
    if wc_acrossclass == 0:
        for i in range(20):
            word_prob[word][i] = 0.05
    else:        
        for i in range(20):
            word_prob[word][i] = word_prob[word][i]/wc_acrossclass

with open('probability_dic.p', 'wb') as handle:
    pickle.dump(word_prob, handle)

