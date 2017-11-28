"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.svm import SVC
if __name__ == '__main__':
    newsgroups_train = fetch_20newsgroups(subset='train')
    targets = newsgroups_train.target
    target_names = newsgroups_train.target_names
    train_data = newsgroups_train.data

    newsgroups_test = fetch_20newsgroups(subset='test')
    test_data = newsgroups_test.data
    test_targets = newsgroups_test.target

    vectorizer = TfidfVectorizer()
    train_X = vectorizer.fit_transform(train_data)
    test_X = vectorizer.transform(test_data)
    # Naiive bayes classifier
    clf = MultinomialNB(alpha=0.01)
    clf.fit(train_X, targets)
    test_preds = clf.predict(test_X)
    print('F1 Score = ', metrics.f1_score(test_targets, test_preds, average='macro'))
    print('Classification accuracy = ', metrics.accuracy_score(test_targets, test_preds))
    print('Classification report:\n', metrics.classification_report(test_targets, test_preds))
