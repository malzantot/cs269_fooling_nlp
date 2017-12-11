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

    def attack(self, x, limit=0.5):
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
        iters_limit = int(limit*np.count_nonzero(x_adv))
        last_predicted = orig_predicted
        changed_words = set()
        while len(changed_words) < (iters_limit):
            word_idx = np.random.choice(len(x_adv[0]))
            # don't perturb paddings
            while x_adv[0][word_idx] == 0:
                word_idx = np.random.choice(len(x_adv[0]))
            changed_words.add(word_idx)
            # select new word
            x_adv[0][word_idx] = np.random.choice(self.num_words)
            adv_scores = self.model.predict(x_adv)
            adv_predicted = np.argmax(adv_scores)
            # print(adv_scores[0][orig_predicted], ' ', np.max(adv_scores[0]))
            pred_history.append(adv_scores[0][orig_predicted])
            last_predicted = np.argmax(adv_scores[0])
            if orig_predicted != adv_predicted:
                # Early stop. Attack done.
                return x_adv, pred_history, last_predicted
        return None, pred_history, last_predicted # Attack failed  

class GreedyAttack(object):
    """ This is a targeted attack that picks preturbed and replacment
    words using a greedy approach """

    def __init__(self, model, topics_words, topics_words_probs, embedding_index, inverse_tokenizer, temp=0.1):
        self.max_iters = 10000
        self.temp = temp
        self.model = model
        self.embedding_index = embedding_index
        self.inverse_tokenizer = inverse_tokenizer
        self.topics_words = topics_words
        self.topics_words_probs = topics_words_probs


    def attack(self, x, target, limit=0.5, use_embedding=False):
        x_adv = x.copy()
        orig_scores = self.model.predict(x_adv)
        orig_predicted = np.argmax(orig_scores)
        orig_history = [orig_scores[0][orig_predicted]]
        target_history = [orig_scores[0][target]]
        target_words = self.topics_words[target]
        target_words_probs = np.array(self.topics_words_probs[target])
        select_logits = np.exp(target_words_probs / self.temp)
        select_probs = select_logits / np.sum(select_logits)
        iters_limit = int(limit * np.count_nonzero(x_adv))
        changed_words = set()
        
        while len(changed_words) < (iters_limit):
            # pick word at random
            word_idx = np.random.choice(len(x_adv[0]))
            # don't perturb paddings
            while x_adv[0][word_idx] == 0:
                word_idx = np.random.choice(len(x_adv[0]))
            changed_words.add(word_idx)
            if use_embedding:
                orig_word_id = x_adv[0][word_idx]
                orig_word = self.inverse_tokenizer[orig_word_id]
                orig_word_vec = self.embedding_index.get(orig_word)
                # select new word
                cand_new_word = np.random.choice(target_words, size=200, p=select_probs)
                if orig_word_vec is None:
                    for w in cand_new_word:
                        if self.embedding_index.get(self.inverse_tokenizer[w]) is None:
                            new_word = w
                            break
                else:
                    min_dist = 1000000000 #infinity
                    min_dist_w = -1
                    for new_w_id in cand_new_word:
                        new_w = self.inverse_tokenizer[new_w_id]
                        new_w_vec = self.embedding_index.get(new_w)
                        if not(new_w_vec is None):
                            # compute dist
                            l2_dist = np.sum((orig_word_vec - new_w_vec)**2)
                            if l2_dist < min_dist:
                                min_dist = l2_dist
                                min_dist_w = new_w_id
                    
                     
                    new_word = min_dist_w    
                        
            else:
                new_word = np.random.choice(target_words, p=select_probs)
            # select with one with minimum distance
            x_adv[0][word_idx] = new_word
            adv_scores = self.model.predict(x_adv)
            orig_history.append(adv_scores[0][orig_predicted])
            target_history.append(adv_scores[0][target])
            adv_predicted = np.argmax(adv_scores[0])
            if adv_predicted == target:
                return x_adv, orig_history, target_history
        return None, orig_history, target_history



            
       
