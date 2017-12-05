"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import numpy as np
import data_utils
import find_most_similar_word
import pickle

class RandomAttack(object):
    def __init__(self, model, num_words):
        self.model = model
        self.num_words = num_words
        self.max_iters = 10000

    def attack(self, x):
        """ Applies attack on input x .
        Returns:
         a tuple of three items:
         x_adv: the modified tokenized document.
         pred_history: the history of original class prediction score.
         last_predicted: the predicted label for x_adv
        """
        x_adv = x.copy()
        orig_scores = self.model.predict(x_adv)
        orig_predicted = np.argmax(orig_scores)
        pred_history = [orig_scores[0][orig_predicted]]
        for iter_idx in range(self.max_iters):
            word_idx = np.random.choice(len(x_adv[0]))
            # don't perturb paddings
            while x_adv[0][word_idx] == 0:
                word_idx = np.random.choice(len(x_adv[0]))
            # select new word
            x_adv[0][word_idx] = np.random.choice(self.num_words)
            adv_scores = self.model.predict(x_adv)
            adv_predicted = np.argmax(adv_scores)
            pred_history.append(adv_scores[0][orig_predicted])
            last_predicted = np.argmax(adv_scores[0])
            if orig_predicted != adv_predicted:
                # Early stop. Attack done.
                return x_adv, pred_history, last_predicted
        return None, pred_history, last_predicted # Attack failed  

class GreedyAttack(object):
    """ This is a targeted attack that picks preturbed and replacment
    words using a greedy approach """

    def __init__(self, model, topics_words, topics_words_probs, temp=0.1):
        self.max_iters = 10000
        self.temp = temp
        self.model = model
        self.topics_words = topics_words
        self.topics_words_probs = topics_words_probs


    def attack(self, x, target):
        x_adv = x.copy()
        tokenizer, inverse_tokenizer = data_utils.load_tokenizer('tokenizer.pickle')
        orig_scores = self.model.predict(x_adv)
        orig_predicted = np.argmax(orig_scores)
        orig_history = [orig_scores[0][orig_predicted]]
        target_history = [orig_scores[0][target]]
        target_words =  [inverse_tokenizer[i] for i in self.topics_words[target]]
        target_words_del = find_most_similar_word.del_words(target_words)
        target_words_probs = np.array(self.topics_words_probs[target])
        select_logits = np.exp(target_words_probs / self.temp)
        select_probs = select_logits / np.sum(select_logits)
        dic = pickle.load(open('probability_dic.p','rb'))
        prob_original = [0 if i == 0 else 0.05 if inverse_tokenizer[i] not in dic else dic[inverse_tokenizer[i]][orig_predicted] for i in x_adv[0]]
        to_replace = sorted(range(len(prob_original)), key=lambda k: prob_original[k], reverse=True)		
        for iter_idx in range(self.max_iters):
            # pick word at random
            #word_idx = np.random.choice(len(x_adv[0]))
            # don't perturb paddings
            #while x_adv[0][word_idx] == 0:
            #    word_idx = np.random.choice(len(x_adv[0]))
            if(iter_idx < len(x_adv[0])) and x_adv[0][to_replace[iter_idx]]!=0:
                word_idx = to_replace[iter_idx]
                if x_adv[0][word_idx] not in inverse_tokenizer:
                    continue
                new_word_temp = find_most_similar_word.most_similar_to_given(inverse_tokenizer[x_adv[0][word_idx]], target_words_del)
                if new_word_temp not in tokenizer.word_index:
                    continue
                new_word = tokenizer.word_index[new_word_temp]
            else:
                new_word = self.topics_words[target][np.random.choice(len(target_words))]
                word_idx = np.random.choice(len(x_adv[0]))
                while x_adv[0][word_idx] == 0:
                    # don't perturb paddings
                    word_idx = np.random.choice(len(x_adv[0]))
            x_adv[0][word_idx] = new_word
            adv_scores = self.model.predict(x_adv)
            orig_history.append(adv_scores[0][orig_predicted])
            target_history.append(adv_scores[0][target])
            adv_predicted = np.argmax(adv_scores)
            if adv_predicted == target:
                return x_adv, orig_history, target_history
        return None, orig_history, target_history



            
       
