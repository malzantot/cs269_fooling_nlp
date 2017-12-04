"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import numpy as np

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
        orig_scores = self.model.predict(x_adv)
        orig_predicted = np.argmax(orig_scores)
        orig_history = [orig_scores[0][orig_predicted]]
        target_history = [orig_scores[0][target]]
        target_words = self.topics_words[target]
        target_words_probs = np.array(self.topics_words_probs[target])
        select_logits = np.exp(target_words_probs / self.temp)
        select_probs = select_logits / np.sum(select_logits)
        for iter_idx in range(self.max_iters):
            # pick word at random
            word_idx = np.random.choice(len(x_adv[0]))
            # don't perturb paddings
            while x_adv[0][word_idx] == 0:
                word_idx = np.random.choice(len(x_adv[0]))
            # select new word
            new_word = np.random.choice(target_words, p=select_probs)
            x_adv[0][word_idx] = new_word
            adv_scores = self.model.predict(x_adv)
            orig_history.append(adv_scores[0][orig_predicted])
            target_history.append(adv_scores[0][target])
            adv_predicted = np.argmax(adv_scores)
            if adv_predicted == target:
                return x_adv, orig_history, target_history
        return None, orig_history, target_history



            
       
