import sys, os
import numpy as np

def compute_topic_words(text_dir, tokenizer, labels, num_words, num_cands=2000, ret_count=200):
    cand_words = np.random.choice(num_words, size=num_cands, replace=False)
    words_probs = np.zeros((num_cands, 20))
    for label_idx in range(len(labels)):
        label_name = labels[label_idx]
        path = os.path.join(text_dir, label_name)
        texts = []
        if os.path.isdir(path):
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
            label_seqs = tokenizer.texts_to_sequences(texts)
            whole_doc = []
            for doc in label_seqs:
                whole_doc.extend(doc)
            whole_doc = np.array(whole_doc)
            for i, w in enumerate(cand_words):
                word_cnt = np.sum(whole_doc==w)
                words_probs[i, label_idx] = word_cnt
    sum_word_probs = np.sum(words_probs, axis=1).reshape((-1,1))
    non_zero_words = np.where(sum_word_probs != 0)[0]
    sum_word_probs = sum_word_probs[non_zero_words]
    words_probs = words_probs[non_zero_words,:]
    cand_words = cand_words[non_zero_words]
    p_words = words_probs / sum_word_probs
    topics_words = []
    topics_words_probs = []
    for idx in range(len(labels)):
        words_list = []
        probs_list = []
        label_probs = p_words[:,idx]
        most_important_words = np.argsort(-label_probs)[:ret_count]
        for w_idx in most_important_words:
            w = cand_words[w_idx]
            words_list.append(w)
            probs_list.append(p_words[w_idx, idx])
        topics_words.append(words_list)
        topics_words_probs.append(probs_list)
    return topics_words, topics_words_probs