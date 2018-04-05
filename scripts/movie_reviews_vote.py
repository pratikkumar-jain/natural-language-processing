import nltk
import pickle
import random

from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from statistics import mode

documents = [(list(movie_reviews.words(fieldid)), category)
             for category in movie_reviews.categories()
             for fieldid in movie_reviews.fileids(category)]

random.shuffle(documents, random=None)

words_list = []

for w in movie_reviews.words():
    words_list.append(w.lower())

words_list = nltk.FreqDist(words_list)

word_feature = list(words_list.keys())[50:5000]


def featuresFind(document):
    feature = {}
    words = set(document)
    for w in word_feature:
        feature[w] = (w in words)

    return feature


featureSet = [(featuresFind(rev), category) for (rev, category) in documents]

trainSet = featureSet[100:1900]
testSet = featureSet[100:]

# trainSet = featureSet[100:1900]
# testSet = featureSet[1900:]

list_of_classifier = [MultinomialNB, LogisticRegression,
                      SGDClassifier, SVC, LinearSVC, NuSVC]
listClassifier = []
for classifier in list_of_classifier:
    classifier_name = SklearnClassifier(classifier())
    listClassifier.append(classifier_name)
    classifier_name.train(trainSet)
    print('{} Accuracy : {}'.format(classifier.__name__,
                                    nltk.classify.accuracy(classifier_name, testSet)*100))


class ClassifierVote(ClassifierI):
    def __init__(self, classifiers):
        self._classifiers = classifiers

    def mode(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)

        vote_choice = votes.count(mode(votes))
        confi = vote_choice / len(votes)
        return confi


votes_classified = ClassifierVote(listClassifier)
for i in range(5):
    print('Confidence : {}'.format(votes_classified.confidence(testSet[i][0])*100))
    print('Mode : {}'.format(votes_classified.mode(testSet[i][0])))
