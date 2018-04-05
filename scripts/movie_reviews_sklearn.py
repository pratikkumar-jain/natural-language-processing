import nltk
import pickle
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents = [(list(movie_reviews.words(fieldid)), category)
             for category in movie_reviews.categories()
             for fieldid in movie_reviews.fileids(category)]

# To print all the documents
# print(documents)

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

trainSet = featureSet[:1400]
testSet = featureSet[1500:]

list_of_classifier = [MultinomialNB, LogisticRegression, SGDClassifier, SVC, LinearSVC, NuSVC]
for classifier in list_of_classifier:
	classifier_name = SklearnClassifier(classifier())
	classifier_name.train(trainSet)
	print('{} Accuracy : {}'.format(classifier.__name__,nltk.classify.accuracy(classifier_name,testSet)*100))


# OUTPUT:
# MultinomialNB Accuracy : 72.4
# LogisticRegression Accuracy : 66.4
# SGDClassifier Accuracy : 66.4
# SVC Accuracy : 51.2
# LinearSVC Accuracy : 63.6
# NuSVC Accuracy : 66.0

