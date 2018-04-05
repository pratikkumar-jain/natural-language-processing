import nltk
import pickle
import random

from nltk.corpus import movie_reviews

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

trainSet = featureSet[:1500]
testSet = featureSet[1500:]

classifys = nltk.NaiveBayesClassifier.train(trainSet)

# to open the classifier saved in the pickle
# open_file = open('nb_classifier.pickle', 'rb')
# classifys = pickle.load(open_file)
# ope
n_file.close()

print("Naive Bayes Accuracy:", (nltk.classify.accuracy(classifys, testSet))*100)
classifys.show_most_informative_features(20)

# use of pickle to save the classifier
# save_file = open('nb_classifier.pickle', 'wb')
# pickle.dump(classifys, save_file)
# save_file.close()

'''
OUTPUT:
('Naive Bayes Accuracy:', 71.39999999999999)
Most Informative Features
               insulting = True              neg : pos    =     12.3 : 1.0
                  doubts = True              pos : neg    =      9.0 : 1.0
              moderately = True              neg : pos    =      7.6 : 1.0
                 wasting = True              neg : pos    =      7.6 : 1.0
                    scum = True              pos : neg    =      7.0 : 1.0
                  quaint = True              pos : neg    =      7.0 : 1.0
             wonderfully = True              pos : neg    =      6.8 : 1.0
              foreboding = True              pos : neg    =      6.4 : 1.0
                    sans = True              neg : pos    =      6.3 : 1.0
              mediocrity = True              neg : pos    =      6.3 : 1.0
             overwhelmed = True              pos : neg    =      5.7 : 1.0
                flawless = True              pos : neg    =      5.6 : 1.0
                   stark = True              pos : neg    =      5.4 : 1.0
              unoriginal = True              neg : pos    =      5.4 : 1.0
                  wasted = True              neg : pos    =      5.2 : 1.0
                   sunny = True              pos : neg    =      5.0 : 1.0
                searches = True              pos : neg    =      5.0 : 1.0
                   lofty = True              pos : neg    =      5.0 : 1.0
                 deadpan = True              pos : neg    =      5.0 : 1.0
                viewings = True              pos : neg    =      5.0 : 1.0
'''
