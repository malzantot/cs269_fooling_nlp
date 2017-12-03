import os
from gensim.scripts.glove2word2vec import glove2word2vec

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')

glove_input_file = os.path.join(GLOVE_DIR, 'glove.6B.300d.txt')
word2vec_output_file = 'glove.6B.300d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
