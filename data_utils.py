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
    inverse_tokenizer = {idx: w for (w, idx) in tokenizer.word_index.items()}
    return tokenizer, inverse_tokenizer

def process_text(tokenizer, text):
    text_seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(text_seq, MAX_SEQUENCE_LENGTH)
    return padded_seq


def reconstruct_text(inverse_tokenizer, x):
    """ Returns the reconstructed text """
    x_words = [inverse_tokenizer[w] for w in x]
    return ' '.join(x_words)

def render_attack(x_orig, x_adv):
    x_orig_words = x_orig.split(' ')
    x_adv_words = x_adv.split(' ')
    orig_html = []
    adv_html = []
    # For now, we assume both original and adversarial text have equal lengths.
    assert(len(x_orig_words) == len(x_adv_words))
    for i in range(len(x_orig_words)):
        if x_orig_words[i] == x_adv_words[i]:
            orig_html.append(x_orig_words[i])
            adv_html.append(x_adv_words[i])
        else:
            orig_html.append(format("<b style='color:green'>%s</b>" %x_orig_words[i]))
            adv_html.append(format("<b style='color:red'>%s</b>" %x_adv_words[i]))
    
    orig_html = ' '.join(orig_html)
    adv_html = ' '.join(adv_html)
    return orig_html, adv_html


