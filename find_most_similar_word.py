
'''
https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
https://radimrehurek.com/gensim/models/keyedvectors.html
'''


# from gensim.scripts.glove2word2vec import glove2word2vec
# glove_input_file = 'glove.6B.100d.txt'
# word2vec_output_file = 'glove.6B.100d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'glove.6B.300d.txt.word2vec'
word_vectors = KeyedVectors.load_word2vec_format(filename, binary=False)


def most_similar(word):
	if word in word_vectors.vocab:
		result = word_vectors.most_similar(positive=[word], topn=1)
		return result[0][0]


def most_similar_to_given(word, word_list):
	if word in word_vectors.vocab:
		for x in word_list: 
			if x not in word_vectors.vocab:
		 		word_list.remove(x)
		return word_vectors.most_similar_to_given(word, word_list)
