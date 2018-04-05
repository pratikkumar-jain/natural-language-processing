import nltk
import pickle
import random

from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from statistics import mode


pos_reviews = open('../short_reviews/positive.txt', 'r').read()
neg_reviews = open('../short_reviews/negative.txt', 'r').read()

documents = []

for words in pos_reviews.split('\n'):
    documents.append(((words), "pos"))

for words in neg_reviews.split('\n'):
    documents.append(((words), "neg"))

words_list = []

pos_reviews_words = word_tokenize(pos_reviews)
neg_reviews_words = word_tokenize(neg_reviews)

for w in pos_reviews_words:
    words_list.append(w)

for wi in neg_reviews_words:
    words_list.append(w)

random.shuffle(documents, random=None)


words_list = nltk.FreqDist(words_list)

word_feature = list(words_list.keys())[50:5000]


def featuresFind(document):
    feature = {}
    words = word_tokenize(document)
    for w in word_feature:
        feature[w] = (w in words)

    return feature


featureSet = [(featuresFind(rev), category) for (rev, category) in documents]

trainSet = featureSet[:9500]
testSet = featureSet[9500:]

list_of_classifier = [MultinomialNB, LogisticRegression,
                      SGDClassifier, SVC, LinearSVC, NuSVC]
listClassifier = []
for classifier in list_of_classifier:
    classifier_name = SklearnClassifier(classifier())
    listClassifier.append(classifier_name)
    classifier_name.train(trainSet)
    print('{} Accuracy : {}'.format(classifier.__name__,
                                    nltk.classify.accuracy(classifier_name, testSet)*100))
