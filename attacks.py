"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""

class RandomAttack(object):
    def __init__(self, model):
        self.model = model
        self.max_iters = 10000

    def attack(self, x):
        x_adv = x.copy()
        orig_scores = model.predict(x_adv)
        orig_predicted = np.argmax(orig_scores)
        for iter_idx in range(self.max_iters):
            word_idx = np.random.choice(len(x_adv[0]))
            # don't perturb paddings
            while x_adv[0][word_idx] == 0:
                word_idx = np.random.choice(len(x_adv[0]))
            # select new word
            x_adv[0][word_idx] = np.random.choice(num_words)
            adv_scores = model.predict(x_adv)
            adv_predicted = np.argmax(adv_scores)
            if orig_predicted != adv_predicted:
                # Early stop. Attack done.
                return x_adv
        return None # Attack failed  




            
       
