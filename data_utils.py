"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import sys
import pickle

MAX_SEQUENCE_LENGTH = 1000

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_textfile(filepath):
    """ Returns the text file words after skipping the header. """
    if sys.version_info < (3,):
        f = open(filepath)
    else:
        f = open(filepath, encoding='latin-1')
    t = f.read()
    i = t.find('\n\n')  # skip header
    if 0 < i:
        t = t[i:]
    return t

def load_tokenizer(tokenizer_path):
    """ Return the tokenizer as keras.preprocessing.text.Tokenizer object """
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def process_text(tokenizer, text):
    text_seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(text_seq, MAX_SEQUENCE_LENGTH)
    return padded_seq






